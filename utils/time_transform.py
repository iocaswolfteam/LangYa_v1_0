import torch
import numpy as np

def time_str2int_list(item, mode='single'):
    """
    Convert a time string to a list of ASCII codes
    Input:
        item:   str     or List[str]
    Output:
        output: ndarray or List[ndarray]
    """
    assert mode in ['single', 'batch'], "mode must be 'single' or 'batch'"
    if mode == 'single':
        return np.array(list(item), dtype='U1').view(np.int32)
    else:
        return np.array([list(item_item) for item_item in item], dtype='U1').view(np.int32)

def int_list2time_str(item, mode='single'):
    """
    Convert a list of ASCII codes to a time string
    Input:
        item: ndarray
    Output:
        output: str
    """
    assert mode in ['single', 'batch'], "mode must be 'single' or 'batch'"
    if mode == 'single':
        if torch.is_tensor(item):
            item = item.cpu().numpy()               # if input is Tensor, convert to ndarray first
        
        char_array = list(item.view('U1'))          # convert int32 array back to character array
        output = ''.join(char_array)                # join character array into a string
    else:
        output = []    
        for i in range(len(item)):
            current_item = item[i]
            if torch.is_tensor(current_item):
                current_item = current_item.cpu().numpy()       # if input is Tensor, convert to ndarray first
            
            # convert to string
            char_array = list(current_item.view('U1'))          # convert int32 array back to character array
            timestamp = ''.join(char_array)                     # join character array into a string
            output.append(timestamp)

    return output

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """Support single string"""
    input_tStr = '19930109'
    target_tStr = '19930110'

    input_tStr = time_str2int_list(input_tStr)
    target_tStr = time_str2int_list(target_tStr)

    input_tStr = torch.tensor(input_tStr, device=device)
    target_tStr = torch.tensor(target_tStr, device=device)

    input_tStr = int_list2time_str(input_tStr, mode='single')
    target_tStr = int_list2time_str(target_tStr, mode='single')

    print("input_tStr: ", input_tStr)
    print("target_tStr: ", target_tStr)

    """Support batch"""
    input_tStr_batch = ['19930101', '19930108', '19930110', '19930109']
    target_tStr_batch = ['19930102', '19930109', '19930111', '19930110']
    # input_tStr_batch = ['19930109']
    # target_tStr_batch = ['19930110']

    input_tStr_batch = time_str2int_list(input_tStr_batch, mode='batch')
    target_tStr_batch = time_str2int_list(target_tStr_batch, mode='batch')

    input_tStr_batch = torch.tensor(input_tStr_batch, device=device)
    target_tStr_batch = torch.tensor(target_tStr_batch, device=device)

    input_tStr_batch = int_list2time_str(input_tStr_batch, mode='batch')
    target_tStr_batch = int_list2time_str(target_tStr_batch, mode='batch')

    print("input_tStr_batch: ", input_tStr_batch)
    print("target_tStr_batch: ", target_tStr_batch)
