import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from utils.time_transform import int_list2time_str
from utils.YParams import YParams
from utils.data_transform import denormalize

from models.LangYa_v1 import LangYa
from datasets.dataset import LangYaDataset


class Tester():
    def __init__(self, params, params_model):
        self.params = params
        self.params_model = params_model
        self.local_rank = self.params['local_rank']
        self.world_rank = self.params['world_rank']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.distributed = self.params['distributed']

        # init dataloader
        self.test_dataset = LangYaDataset(self.params, train=False)
        self.print_log("test_dataset num=", len(self.test_dataset))

        self.test_sampler = DistributedSampler(self.test_dataset) if self.distributed else None
            
        self.test_dataloader = DataLoader(self.test_dataset, 
                                          batch_size=int(self.params['batch_size']), 
                                          num_workers=self.params['num_data_workers'],
                                          shuffle=False,
                                          sampler=self.test_sampler)

        # init model
        self.model = LangYa(self.params_model)
        self.model = self.model.cuda()

        # init air drive weight
        self.air_weight = nn.Parameter(torch.randn(10) * 0.01)
        self.air_weight_softmax = nn.Softmax(dim=0)

        if self.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=[self.local_rank], find_unused_parameters=True)
        
        # load weights
        self.checkpoint_path = self.params['weight_path']
        self.print_log("Loading checkpoint %s ..." % self.checkpoint_path)
        self.restore_checkpoint(self.checkpoint_path)
        self.print_log("Loading weights successfully from %s !!!" % self.checkpoint_path)

        # save dir
        self.save_dir = os.path.join(self.params['work_dir'], 'test_results')
        os.makedirs(self.save_dir, exist_ok=True)

    def test(self):
        self.print_log("Starting Testing Loop...")
        
        self.model.eval()
        with torch.no_grad():
            if self.world_rank == 0:
                tqdm_iter = tqdm(self.test_dataloader, desc="Testing")
            else:
                tqdm_iter = self.test_dataloader
            for data in tqdm_iter:
                # read data
                inp, inp_air, tar, input_times, _, _ = data
                inp = inp.cuda().float()
                inp_air = inp_air.cuda().float()
                tar = tar.cuda().float()
                input_times = input_times.cuda()
                input_times_str = int_list2time_str(input_times[0], mode='single')

                # calculate air drive weight
                air_weight = self.air_weight_softmax(self.air_weight)
                air_weight = air_weight.view(1, 10, 1, 1, 1)
                inp_air_weight = inp_air * air_weight.cuda().float()
                inp_air_weight = inp_air_weight.sum(dim=1)
                inps = torch.cat((inp, inp_air_weight), dim=1)
                
                # forward process
                if not self.params_model['use_time_embed']:
                    gen = self.model(inps)
                else:
                    gen = self.model(inps, input_times, torch.tensor(1))
                
                # denormalize
                gen = denormalize(self.params, gen.detach().cpu().numpy())
                tar = denormalize(self.params, tar.detach().cpu().numpy())
                gen[np.isnan(tar)] = np.nan

                # save results as npy
                save_path = os.path.join(self.save_dir, f'{input_times_str}_1.npy')
                np.save(save_path, gen)

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)
        self.print_log("Restore model weights from %s" % checkpoint_path)
        if 'air_weight' in checkpoint:
            self.air_weight.data = checkpoint['air_weight'].data
            self.print_log("Restore air weights from %s" % checkpoint_path)
        else:
            self.print_log("Warning: No air weights in %s" % checkpoint_path)
    
    def print_log(self, *values):
        if self.world_rank == 0:
            print(*values)


def main(args):
    # read config file
    with open(args.config_path, 'r'):
        params = YParams(os.path.abspath(args.config_path), 'full_field')
    with open(args.config_path, 'r'):
        params_model = YParams(os.path.abspath(args.config_path), 'all_model')

    # set weight path
    params['weight_path'] = args.weight_path
    
    # set distributed parameters
    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])
    else:
        print("else:world_size=1")

    world_rank = 0
    local_rank = 0
    if params['world_size'] > 1:
        dist.init_process_group(backend='nccl',
                                init_method='env://')
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()
        params['batch_size'] = int(os.environ['TOTAL_BATCH_SIZE']) // params['world_size']
        params['global_batch_size'] = int(os.environ['TOTAL_BATCH_SIZE'])

    params['local_rank'] = local_rank
    params['world_rank'] = world_rank
    print("local rank=", params['local_rank'], "world_rank=", params['world_rank'])
    print("world_size=", params['world_size'])

    torch.cuda.set_device(local_rank)

    distributed = dist.is_initialized()
    assert distributed == params['distributed']

    # set work dir
    params['work_dir'] = os.path.join(args.work_dir, 'langya_test_results')
    if params['world_rank'] == 0:
        os.makedirs(params['work_dir'], exist_ok=True)

    # start testing
    torch.backends.cudnn.benchmark = True
    tester = Tester(params, params_model)
    tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='configs/langya_test_config.yaml', type=str)
    parser.add_argument("--weight_path", default='weights/langya_v1.tar', type=str)
    parser.add_argument("--work_dir", default='work_dirs', type=str)

    args = parser.parse_args()
    main(args)
