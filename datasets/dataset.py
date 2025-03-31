import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import random
from datetime import datetime
from collections import OrderedDict
from scipy.ndimage import zoom
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root directory to python path
from utils.data_transform import preprocess_all_data
from utils.time_transform import time_str2int_list
T_FORMAT='%Y%m%d'


class LangYaDataset(Dataset):
    def __init__(self, params, train=True, num_files=None):
        """Dataset class for loading and preprocessing meteorological data. Supports both single-step and multi-step prediction modes.

        Args:
            - params (YParams): Configuration parameter object containing:
                - location_*: Storage paths for different data types (sst/atmosphere/ocean/precip)
                - resolution_*: Target resolution
                - forecast_day: Number of days to forecast [only used when not in training stage]
                - n_history: Number of historical days
                - n_history_air: Number of atmospheric historical days
                - flag_single_across: Whether in single-step prediction mode, means predict one day ahead
                - max_random_step_training: Maximum random step for training
                - *_in_channels: Input channels for each data type
                - *_out_channels: Output channels for each data type
                - orography: Whether to use orography
                - add_noise: Whether to add noise
                - roll: Whether to roll
                
            - train (bool, optional): Whether in training stage. Defaults to True.
            - num_files (int, optional): Limit on number of files to load. Defaults to None.

        Data Processing Flow:
            1. Determine input/output sequences based on prediction mode (single/multi-step)
            2. Load and preprocess data (SST/atmosphere/ocean etc.)
            3. Process timestamp information
            4. Apply spatial transformations (interpolation, cropping etc.)
            5. Convert to PyTorch tensor format

        Returns:
            Each __getitem__ call returns:
                - input_tsor: Input data tensor [B, C, H, W]
                - input_tsor_atmosphere: Atmospheric input tensor [B, C, H, W]
                - target_tsor: Target data tensor [B, C, H, W]
                - input_times: Input timestamps
                - target_times: Target timestamps
                - step_times: Prediction step lengths
        """
        # Initialize parameters
        self.params = params
        self.location_sst = params.location_sst
        self.location_atmosphere = params.location_atmosphere
        self.location_ocean = params.location_ocean
        self.location_precip = params.location_precip

        self.location_sst_valid = params.location_sst_valid
        self.location_atmosphere_valid = params.location_atmosphere_valid
        self.location_ocean_valid = params.location_ocean_valid
        self.location_precip_valid = params.location_precip_valid

        self.resolution_y = params.resolution_y
        self.resolution_x = params.resolution_x

        self.train = train
        self.forecast_day = params.forecast_day
        self.n_history = params.n_history
        self.n_history_air = params.n_history_air

        self.flag_single_across = params.flag_single_across
        self.max_random_step_training = params.max_random_step_training

        self.ocean_in_channels = params.ocean_in_channels
        self.ocean_out_channels = params.ocean_out_channels
        self.sst_in_channels = params.sst_in_channels
        self.sst_out_channels = params.sst_out_channels
        self.atmosphere_in_channels = params.atmosphere_in_channels
        self.atmosphere_out_channels = params.atmosphere_out_channels

        # Initialize other data preprocessing parameters
        self.add_noise = params.add_noise if train else False
        self.roll = params.roll
        self.normalize = params.normalize if 'normalize' in params else True
        self.orography = params.orography
        if self.orography:
            self.orography_path = params.orography_path

        # Get file paths and statistics
        self.num_files = num_files
        self._get_files_stats()
        self.n_samples_total = self.n_days

    def _get_files_stats(self):
        """Get file paths and statistics for the dataset.
        
        This method:
        1. Gets paths for all data types (SST/atmosphere/ocean/precip) based on training/validation/test stage
        2. Sorts file paths by basename to ensure temporal order
        3. Limits number of files if specified
        4. Gets image shape information from first ocean file
        5. Initializes file lists for lazy loading
        
        File Structure:
            - Training files: location_*/
            - Validation files: location_*_valid/
            Each location contains .npy files with shape [T, C, H, W]
            
        Sets the following attributes:
            - *_paths: Lists of file paths for each data type
            - n_days: Total number of days in dataset
            - img_shape_x/y: Spatial dimensions of data
            - *_files: Lists for lazy loading of files
        """
        # Get file paths based on training/validation/test stage
        if self.train:
            self.sst_paths = glob.glob(self.location_sst + "/*.npy") if self.location_sst else []
            self.atmosphere_paths = glob.glob(self.location_atmosphere + "/*.npy") if self.location_atmosphere else []
            self.ocean_paths = glob.glob(self.location_ocean + "/*.npy") if self.location_ocean else []
            self.precip_paths = glob.glob(self.location_precip + "/*.npy") if self.location_precip else []
        else:
            self.sst_paths = glob.glob(self.location_sst_valid + "/*.npy") if self.location_sst_valid else []
            self.atmosphere_paths = glob.glob(
                self.location_atmosphere_valid + "/*.npy") if self.location_atmosphere_valid else []
            self.ocean_paths = glob.glob(self.location_ocean_valid + "/*.npy") if self.location_ocean_valid else []
            self.precip_paths = glob.glob(self.location_precip_valid + "/*.npy") if self.location_precip_valid else []

        # Sort file paths by basename to ensure temporal order
        if self.sst_paths:
            self.sst_paths.sort(key=os.path.basename)
        if self.atmosphere_paths:
            self.atmosphere_paths.sort(key=os.path.basename)
        if self.ocean_paths:
            self.ocean_paths.sort(key=os.path.basename)
        if self.precip_paths:
            self.precip_paths.sort(key=os.path.basename)

        # Limit number of files if specified
        if self.num_files:
            self.sst_paths = self.sst_paths[:self.num_files]
            self.atmosphere_paths = self.atmosphere_paths[:self.num_files]
            self.ocean_paths = self.ocean_paths[:self.num_files]
            self.precip_paths = self.precip_paths[:self.num_files]

        # Get image shape information from first ocean file
        if self.ocean_paths:
            self.n_days = len(self.ocean_paths)
            f = np.load(self.ocean_paths[0])
            self.img_shape_x = f.shape[2]  # 2041
            self.img_shape_y = f.shape[3]  # 4320
        else:
            self.n_days = 0
            self.img_shape_x = 680
            self.img_shape_y = 1440

        # Initialize file lists for lazy loading
        if self.location_sst:
            self.sst_files = [None for _ in range(self.n_days)]
        if self.location_atmosphere:
            self.atmosphere_files = [None for _ in range(self.n_days)]
        if self.location_ocean:
            self.ocean_files = [None for _ in range(self.n_days)]
    
    def __getitem__(self, global_idx):
        """Get a sample from the dataset.

        Args:
            global_idx (int): Index of the current sample.

        Returns:
            tuple: A tuple containing:
                - input_tsor (Tensor): Input data tensor [B, C, H, W]
                - input_tsor_atmosphere (Tensor): Atmospheric input tensor [B, C, H, W]
                - target_tsor (Tensor): Target data tensor [B, C, H, W]
                - input_times (Tensor): Input timestamps
                - target_times (Tensor): Target timestamps
                - step_times (Tensor): Prediction step lengths
        """

        """I. Get required data sequences indices"""
        if self.flag_single_across:
            """1. Determine prediction step length"""
            # For training stage: Random step between 1 and max_random_step_training (default 1-7 days)
            # For validation/testing stage: Fixed step length (default 1 day)
            if self.train:
                self.step = random.randint(1, self.max_random_step_training)
            else:
                self.step =  self.forecast_day
            
            """2. Determine input and output sequences"""
            # Input: Current time and past n_history days. Default n_history = 0
            # Output: Single frame prediction for a future day, determined by step
            input_idx = list(range(global_idx - self.n_history, global_idx + 1))
            output_idx = list(range(global_idx + self.step, global_idx + self.step + 1))
            
            # Atmospheric input: Past n_history_air days of atmospheric data
            # e.g., for past 10 days: t=-9,-8,-7,-6,-5,-4,-3,-2,-1,0
            air_input_idx = list(range(global_idx - self.n_history_air + 1, global_idx + 1))
            
            """3. Handle boundary conditions for input/output sequences"""
            # 3.1 When global_idx is smaller than n_history
            # Handle negative indices for the first few samples when n_history != 0
            # Special handling for atmospheric data due to n_history_air requirement
            if global_idx < (self.n_history + self.n_history_air):
                input_idx = list(range(global_idx, global_idx + self.n_history + 1))
                air_input_idx = list(range(0, self.n_history_air))
                output_idx = list(range(global_idx + self.n_history + self.step, global_idx + self.n_history + self.step + 1))
            
            # 3.2 When global_idx + step exceeds maximum index (n_days)
            # Handle index overflow for the last few samples
            if (global_idx + self.step + 1) > self.n_days:
                input_idx = list(range(self.n_days - self.step - self.n_history - 1, self.n_days - self.step))
                air_input_idx = list(range(global_idx - self.step - self.n_history_air - 1, self.n_days - self.step))[0:self.n_history_air]
                output_idx = list(range(self.n_days - 1, self.n_days))
        else:
            raise ValueError("multi-step prediction is not supported yet.")
        
        """II. Load required data sequences"""
        # 1. Initialize lists for different data types
        if self.location_sst:
            input_sst_list, output_sst_list = [], []

        if self.location_atmosphere:
            input_atmosphere_list, output_atmosphere_list = [], []

        if self.location_ocean:
            input_ocean_list, output_ocean_list = [], []

        # 2. Load data and store in lists
        input_times = OrderedDict()
        target_times = OrderedDict()
        # 2.1 Load input data
        for idx in input_idx:
            if (self.location_sst and self.sst_files[idx] is None) or \
                    (self.location_atmosphere and self.atmosphere_files[idx] is None) or \
                    (self.location_ocean and self.ocean_files[idx] is None):
                self._map_file_paths(idx)

            if self.location_sst:
                input_sst_data = self._transform_data(np.load(self.sst_files[idx]).astype(np.float32),
                                                      (self.img_shape_x, self.img_shape_y),
                                                      self.sst_in_channels, interpolation=True)
                input_sst_list.append(input_sst_data)

                # Record timestamp, mark as 1 for time embedding
                time_key = self.sst_files[idx].split(".")[-2]
                input_times[time_key] = 1

            if self.location_ocean:
                input_ocean_data = self._transform_data(np.load(self.ocean_files[idx]).astype(np.float32),
                                                        (self.img_shape_x, self.img_shape_y),
                                                        self.ocean_in_channels, interpolation=False)  # ocean data is no need to be interpolated
                input_ocean_list.append(input_ocean_data)

                # Record timestamp, mark as 1 for time embedding
                time_key = self.ocean_files[idx].split("_")[-2]
                input_times[time_key] = 1
        
        # 2.2 Load atmospheric input data
        for idx in air_input_idx:
            if (self.location_atmosphere and self.atmosphere_files[idx] is None):
                self._map_file_paths(idx)
            if self.location_atmosphere:
                input_atmosphere_data = self._transform_data(np.load(self.atmosphere_files[idx]).astype(np.float32),
                                                            (self.img_shape_x, self.img_shape_y),
                                                            self.atmosphere_in_channels, interpolation=True)
                input_atmosphere_list.append(input_atmosphere_data)

        # 2.3 Load output data
        for idx in output_idx:
            if (self.location_sst and self.sst_files[idx] is None) or \
                    (self.location_atmosphere and self.atmosphere_files[idx] is None) or \
                    (self.location_ocean and self.ocean_files[idx] is None):
                self._map_file_paths(idx)

            if self.location_sst:
                output_sst_data = self._transform_data(np.load(self.sst_files[idx]).astype(np.float32),
                                                       (self.img_shape_x, self.img_shape_y),
                                                       self.sst_in_channels, interpolation=True)
                output_sst_list.append(output_sst_data)

                time_key = self.sst_files[idx].split(".")[-2]
                target_times[time_key] = 1

            if self.location_atmosphere:
                output_atmosphere_data = self._transform_data(np.load(self.atmosphere_files[idx]).astype(np.float32),
                                                              (self.img_shape_x, self.img_shape_y),
                                                              self.atmosphere_in_channels, interpolation=True)
                output_atmosphere_list.append(output_atmosphere_data)

            if self.location_ocean:
                output_ocean_data = self._transform_data(np.load(self.ocean_files[idx]).astype(np.float32),
                                                         (self.img_shape_x, self.img_shape_y),
                                                         self.ocean_in_channels, interpolation=False)  # ocean data is no need to be interpolated
                output_ocean_list.append(output_ocean_data)

                time_key = self.ocean_files[idx].split("_")[-2]
                target_times[time_key] = 1

        """III. Process data"""
        # 1. Get timestamp lists
        input_times = list(input_times.keys())                                                                              # List[str]
        target_times = list(target_times.keys())                                                                            # List[str]
        step_times = [(datetime.strptime(tar_t, T_FORMAT).date() - datetime.strptime(in_t, T_FORMAT).date()).days 
                      for in_t, tar_t in zip(input_times, target_times)]                                                    # List[int]

        if self.flag_single_across:
            # Simplify for subsequent conversion logic
            input_times = input_times[0]                                                                                    # str
            target_times = target_times[0]                                                                                  # str
            step_times = step_times[0]                                                                                      # int

            # Convert timestamps to ASCII code lists for tensor conversion
            input_times = time_str2int_list(input_times, mode='single')                                                     # ndarray
            target_times = time_str2int_list(target_times, mode='single')                                                   # ndarray
        else:
            # Convert timestamps to ASCII code lists for tensor conversion
            input_times = time_str2int_list(input_times, mode='batch')                                                      # List[ndarray]
            target_times = time_str2int_list(target_times, mode='batch')                                                    # List[ndarray]
        
        # 2. Concatenate all data along time dimension -> 1*[T C H W] -> [n*T C H W]
        input_data_list, output_data_list = [], []
        if self.location_sst:
            input_sst = np.concatenate(input_sst_list, axis=0)                                                              # [T C H W]
            output_sst = np.concatenate(output_sst_list, axis=0)                                                            # [T C H W]
            input_data_list.append(input_sst)
            output_data_list.append(output_sst)

        if self.location_atmosphere:
            input_atmosphere = np.concatenate(input_atmosphere_list, axis=0)                                                # n_history_air*[T C H W] -> [n_history_air*T C H W]
            output_atmosphere = np.concatenate(output_atmosphere_list, axis=0)                                              # [T C H W]

        if self.location_ocean:
            input_ocean = np.concatenate(input_ocean_list, axis=0)                                                          # [T C H W]
            output_ocean = np.concatenate(output_ocean_list, axis=0)                                                        # [T C H W]
            input_data_list.append(input_ocean)
            output_data_list.append(output_ocean)

        # 3. Concatenate all data along channel dimension, each channel represents different variables
        input_tsor = np.concatenate(input_data_list, axis=1)
        target_tsor = np.concatenate(output_data_list, axis=1)

        # 4. Get orography data if exists
        if self.orography:
            raise ValueError("orography is not supported yet.")
        else:
            orog = None

        """IV. Preprocess all data"""
        # Initialize parameters for cyclic shift and random cropping
        if self.roll == False:
            y_roll = 0                  # Distance for y-axis cyclic shift
            rnd_x = 0                   # Starting x position for cropping
            rnd_y = 0                   # Starting y position for cropping
            self.crop_size_x = None     # Crop size along x-axis
            self.crop_size_y = None     # Crop size along y-axis
        else:
            raise ValueError("roll and random crop is not supported yet.")

        # Preprocess input and target data
        if self.location_atmosphere:
            input_tsor, input_tsor_atmosphere = preprocess_all_data(
                img=input_tsor, 
                img_atmosphere=input_atmosphere, 
                inp_or_tar='inp',
                crop_size_x=self.crop_size_x, 
                crop_size_y=self.crop_size_y,
                rnd_x=rnd_x,
                rnd_y=rnd_y,
                params=self.params,
                y_roll=y_roll,
                normalize=self.normalize,
                orog=orog,
                add_noise=self.add_noise
            )
            
            target_tsor = preprocess_all_data(
                img=target_tsor,
                img_atmosphere=output_atmosphere,
                inp_or_tar='tar',
                crop_size_x=self.crop_size_x,
                crop_size_y=self.crop_size_y,
                rnd_x=rnd_x,
                rnd_y=rnd_y,
                params=self.params,
                y_roll=y_roll,
                normalize=self.normalize,
                orog=orog,
                add_noise=self.add_noise
            )

        else:
            input_tsor, input_tsor_atmosphere = preprocess_all_data(
                img=input_tsor,
                img_atmosphere=input_atmosphere,
                inp_or_tar='inp',
                crop_size_x=self.crop_size_x,
                crop_size_y=self.crop_size_y,
                rnd_x=rnd_x,
                rnd_y=rnd_y,
                params=self.params,
                y_roll=y_roll,
                normalize=self.normalize,
                orog=orog,
                add_noise=self.add_noise
            )
            
            target_tsor = preprocess_all_data(
                img=target_tsor,
                img_atmosphere=output_atmosphere,
                inp_or_tar='tar',
                crop_size_x=self.crop_size_x,
                crop_size_y=self.crop_size_y,
                rnd_x=rnd_x,
                rnd_y=rnd_y,
                params=self.params,
                y_roll=y_roll,
                normalize=self.normalize,
                orog=orog,
                add_noise=self.add_noise
            )

        """V. Convert to tensors and apply spatial interpolation"""
        # Convert to tensors and apply spatial interpolation
        input_tsor = torch.from_numpy(self.interpolate(input_tsor, (self.resolution_y, self.resolution_x))).float()
        target_tsor = torch.from_numpy(self.interpolate(target_tsor, (self.resolution_y, self.resolution_x))).float()

        input_tsor_atmosphere = torch.from_numpy(self.interpolate(input_tsor_atmosphere, (self.resolution_y, self.resolution_x))).float()
        input_tsor_atmosphere = input_tsor_atmosphere[0:self.n_history_air,:,:,:] # to ensure only the first n_history_air days are used
        input_times = torch.tensor(input_times)
        target_times = torch.tensor(target_times)
        step_times = torch.tensor(step_times)

        return input_tsor, input_tsor_atmosphere, target_tsor, input_times, target_times, step_times   # (B, C, H, W), (B, C, H, W), (B, T), (B, T), (B)

    def _map_file_paths(self, idx):
        """Lazy loading helper function to load data files at specific index.
        
        This method implements lazy loading by only storing file paths until they are needed.
        When a file is requested, it maps the file path to the corresponding file list
        for each data type (SST/atmosphere/ocean).

        Args:
            idx (int): Index of the day to load data for. Represents the idx'th day in the dataset.

        Note:
            - This method doesn't actually load the file content, it only stores the file path
            - Actual file loading happens when the data is accessed in _transform_data
            - This approach saves memory by avoiding loading all files at once
        """
        if self.location_sst:
            self.sst_files[idx] = self.sst_paths[idx]

        if self.location_atmosphere:
            self.atmosphere_files[idx] = self.atmosphere_paths[idx]

        if self.location_ocean:
            self.ocean_files[idx] = self.ocean_paths[idx]
    
    def _transform_data(self, data, target_shape, channels, interpolation=False):
        """Process and transform the input data.
        
        This method performs several data processing steps:
        1. Select specified channels from input data
        2. Expand 3D data to 4D with batch size 1
        3. Apply spatial transformations if interpolation is True:
            - Reverse height dimension
            - Swap width halves (for handling data across the international date line)
            - Interpolate to target shape
        
        Args:
            data (np.ndarray): Input data with shape [T, C, H, W]
            target_shape (tuple): Target spatial dimensions (H, W)
            channels (list): Indices of channels to select
            interpolation (bool, optional): Whether to apply spatial transformations. 
                Defaults to False.
        
        Returns:
            np.ndarray: Processed data with shape [T, C_selected, H_new, W_new]
        """
        datas = data[:, channels, :, :]

        # Expand 3D data to 4D with batch size 1
        if data.ndim < 4:
            data = np.expand_dims(data, 0)

        if interpolation:
            datas = self.reverse_height_dimension(datas, (self.img_shape_x, self.img_shape_y))
            datas = self.swap_width_halves(datas, (self.img_shape_x, self.img_shape_y))
            datas = self.interpolate(datas, target_shape)

        return datas

    def reverse_height_dimension(self, data, target_shape):
        """Reverse the height dimension of the input data if shape doesn't match target.
        
        This function performs two operations on the height dimension:
        1. Crops the height to 681 pixels
        2. Flips the height dimension vertically
        These operations are only performed if the input shape doesn't match the target shape.
        
        Args:
            data (np.ndarray): Input data with shape [T, C, H, W].
                If 3D, will be expanded to 4D with batch size 1.
            target_shape (tuple): Target spatial dimensions (H, W).
                Default is (2041, 4320).
        
        Returns:
            np.ndarray: Processed data with reversed height dimension if needed.
                Shape remains [T, C, H, W].
        """

        # Get input data shape
        _, _, H, W = data.shape

        # Check if shape matches target
        if (H, W) != target_shape:
            # Crop and reverse height dimension
            data = data[:, :, 0:681, :]
            data = data[:, :, ::-1, :]

        return data

    def swap_width_halves(self, data, target_shape):
        """Swap the left and right halves of the data along the width dimension.
        
        This function handles data that crosses the international date line by:
        1. Splitting the width dimension at the midpoint
        2. Swapping the left and right halves
        This operation is only performed if the input shape doesn't match the target shape.
        
        Args:
            data (np.ndarray): Input data with shape [T, C, H, W].
                If 3D, will be expanded to 4D with batch size 1.
            target_shape (tuple): Target spatial dimensions (H, W).
                Default is (2041, 4320).
        
        Returns:
            np.ndarray: Processed data with swapped width halves if needed.
                Shape remains [T, C, H, W].
        
        Note:
            This transformation is used to handle meteorological data that crosses
            the international date line (180° longitude), ensuring spatial continuity
            in the data representation.
        """

        # Get input data shape
        _, _, H, W = data.shape
        # Check if shape matches target
        if (H, W) != target_shape:
            # Calculate midpoint of width dimension
            mid_point = W // 2

            # Swap left and right halves of the data
            data = np.concatenate((data[:, :, :, mid_point:], data[:, :, :, :mid_point]), axis=3)

        return data

    def interpolate(self, data, target_shape, method='nearest'):
        """Interpolate data to target spatial dimensions.
        
        This function resizes the spatial dimensions (H, W) of the input data while
        preserving other dimensions. It automatically handles different input dimensions:
        - 3D input [C, H, W]: Used for single-frame data
        - 4D input [T, C, H, W]: Used for multi-frame data
        
        Args:
            data (np.ndarray): Input data with shape [C, H, W] or [T, C, H, W].
            target_shape (tuple): Target spatial dimensions [H_new, W_new].
                If length is 3: [C_new, H_new, W_new] - also resize channel dimension
                (only supported for 4D input).
            method (str, optional): Interpolation method to use.
                Options:
                    - 'nearest': Nearest neighbor interpolation
                    - 'bilinear': Bilinear interpolation
                    - 'bicubic': Bicubic interpolation
                Defaults to 'nearest'.
        
        Returns:
            np.ndarray: Interpolated data with shape matching input dimensions:
                - 3D input -> [C, H_new, W_new]
                - 4D input -> [T, C, H_new, W_new]
        
        Raises:
            ValueError: If interpolation method is invalid or if target_shape is incompatible.
        """
        # Map interpolation methods to scipy.ndimage.zoom orders
        method_map = {
            'nearest': 0,
            'bilinear': 1,
            'bicubic': 2
        }

        if method not in method_map:
            raise ValueError("Invalid interpolation method. Choose from 'nearest', 'bilinear', 'bicubic'.")

        # Handle different input dimensions
        if data.ndim == 3:  # [C, H, W]
            if len(target_shape) != 2:
                raise ValueError("For 3D input, target_shape must be [H_new, W_new]")
            zoom_factors = [1, target_shape[0] / data.shape[1], target_shape[1] / data.shape[2]]
        
        elif data.ndim == 4:  # [T, C, H, W]
            if len(target_shape) == 2:
                zoom_factors = [1, 1, target_shape[0] / data.shape[2], target_shape[1] / data.shape[3]]
            elif len(target_shape) == 3:
                zoom_factors = [1, target_shape[0] / data.shape[1], 
                            target_shape[1] / data.shape[2],
                            target_shape[2] / data.shape[3]]
            else:
                raise ValueError("For 4D input, target_shape must be [H_new, W_new] or [C_new, H_new, W_new]")
        
        else:
            raise ValueError(f"Unsupported input dimensions: {data.ndim}. Must be 3 or 4.")

        return zoom(data, zoom_factors, order=method_map[method])

    def __len__(self):
        return self.n_samples_total - 1

