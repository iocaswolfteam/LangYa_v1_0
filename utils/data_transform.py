import numpy as np


def preprocess_all_data(img, img_atmosphere, inp_or_tar, 
                   crop_size_x, crop_size_y, 
                   rnd_x, rnd_y, 
                   params, 
                   y_roll, 
                   normalize=False, 
                   orog=None, add_noise=False):
    """
    Function to reshape and preprocess input data.

    preprocess the input data, including:
    - normalization
    - orography
    - random y-axis shift
    - random cropping
    - reshaping
    
    Args:
        img: Input image data with shape (n_history+1, channels, height, width)
        img_atmosphere: Atmospheric data
        inp_or_tar: String, 'inp' for input data, 'tar' for target data
        crop_size_x: Target size after cropping along x-axis
        crop_size_y: Target size after cropping along y-axis
        rnd_x: Starting position for random cropping along x-axis
        rnd_y: Starting position for random cropping along y-axis
        params: Configuration object containing parameters
        y_roll: Amount of cyclic shift along y-axis
        normalize: Whether to normalize data, defaults to False
        orog: Orography data, defaults to None
        add_noise: Whether to add random noise, defaults to False
        
    Returns:
        Processed data with shape (channels, crop_size_x, crop_size_y)
    """

    """1. Ensure input image is 4D"""
    #  (n_history+1, channels, height, width)
    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)

    """2. Crop data to specified y resolution"""
    img = img[:, :, 0:params.resolution_y, :]                           # remove last pixel
    img_atmosphere = img_atmosphere[:, :, 0:params.resolution_y, :]     # remove last pixel

    """3. Get data shapes"""
    _, n_channels, img_shape_x, img_shape_y = img.shape

    """4. Initialize cropping parameters"""
    crop_size_x = img_shape_x if crop_size_x is None else crop_size_x
    crop_size_y = img_shape_y if crop_size_y is None else crop_size_y

    """5. Normalize data (follow the normalization method in minmax or zscore)"""
    if normalize:
        start_idx = 0
        norm_method = params.normalization  # 'minmax' or 'zscore'
        
        # Normalize SST data
        if params.location_sst:
            stat1, stat2, n_channels = _load_stats(params, 'sst', inp_or_tar, norm_method)
            end_idx = start_idx + n_channels
            img[:, start_idx:end_idx] = _normalize(img[:, start_idx:end_idx], stat1, stat2, norm_method)
            start_idx = end_idx
        
        # Normalize ocean data
        if params.location_ocean:
            stat1, stat2, n_channels = _load_stats(params, 'ocean', inp_or_tar, norm_method)
            end_idx = start_idx + n_channels
            img[:, start_idx:end_idx] = _normalize(img[:, start_idx:end_idx], stat1, stat2, norm_method)
            start_idx = end_idx
        
        # Normalize atmosphere data
        if params.location_atmosphere:
            stat1, stat2, _ = _load_stats(params, 'atmosphere', inp_or_tar, norm_method)
            img_atmosphere = _normalize(img_atmosphere, stat1, stat2, norm_method)
        
        # Handle NaN values for input data
        if inp_or_tar == 'inp':
            img[np.isnan(img)] = 0
            img_atmosphere[np.isnan(img_atmosphere)] = 0

    """6. Grid data"""
    if params.add_grid:
        if inp_or_tar == 'inp':
            if params.gridtype == 'linear':
                assert params.N_grid_channels == 2, "N_grid_channels must be set to 2 for gridtype linear"
                x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
                y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
                grid_x, grid_y = np.meshgrid(y, x)
                grid = np.stack((grid_x, grid_y), axis=0)
            elif params.gridtype == 'sinusoidal':
                assert params.N_grid_channels == 4, "N_grid_channels must be set to 4 for gridtype sinusoidal"
                x1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, img_shape_x)))
                x2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, img_shape_x)))
                y1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, img_shape_y)))
                y2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, img_shape_y)))
                grid_x1, grid_y1 = np.meshgrid(y1, x1)
                grid_x2, grid_y2 = np.meshgrid(y2, x2)
                grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis=0), axis=0)
            img = np.concatenate((img, grid), axis=1)

    """7. Add orography data"""
    if params.orography and inp_or_tar == 'inp':
        img = np.concatenate((img, np.expand_dims(orog, axis=(0, 1))), axis=1)
        n_channels += 1

    """8. Random y-axis cyclic shift"""
    if params.roll:
        img = np.roll(img, y_roll, axis=-1)
        img_atmosphere = np.roll(img_atmosphere, y_roll, axis=-1)

    """9. Random cropping"""
    if crop_size_x or crop_size_y:
        img = img[:, :, rnd_x:rnd_x + crop_size_x, rnd_y:rnd_y + crop_size_y]
        img_atmosphere = img_atmosphere[:, :, rnd_x:rnd_x + crop_size_x, rnd_y:rnd_y + crop_size_y]

    """10. Reshape data"""
    if inp_or_tar == 'inp':
        img = np.reshape(img, (img.shape[0] * img.shape[1], crop_size_x, crop_size_y))
            
    elif inp_or_tar == 'tar':
        img = np.reshape(img, (img.shape[0] * img.shape[1], crop_size_x, crop_size_y))

    """11. Add noise"""
    if add_noise:
        img = img + np.random.normal(0, scale=params.noise_std, size=img.shape)
    
    """12. Return data"""
    if inp_or_tar == 'tar':
        return img
    else:    
        return img, img_atmosphere

def denormalize(params, img, inp_or_tar='inp'):
    """Denormalize the input data.
    
    Args:
        params: Parameter object containing configuration
        img: Input data to denormalize
        inp_or_tar: String, 'inp' for input data or 'tar' for target data
    
    Returns:
        Denormalized data
    """
    if params.normalization == 'zscore':
        start_idx = 0
        
        # Denormalize SST data
        if params.location_sst:
            means, stds, n_channels = _load_stats(params, 'sst', inp_or_tar, 'zscore')
            end_idx = start_idx + n_channels
            img[:, start_idx:end_idx] = _denormalize(img[:, start_idx:end_idx], means, stds)
            start_idx = end_idx
        
        # Denormalize ocean data
        if params.location_ocean:
            means, stds, n_channels = _load_stats(params, 'ocean', inp_or_tar, 'zscore')
            end_idx = start_idx + n_channels
            img[:, start_idx:end_idx] = _denormalize(img[:, start_idx:end_idx], means, stds)
            start_idx = end_idx
    
    elif params.normalization == 'minmax':
        raise NotImplementedError("Minmax denormalization not implemented yet")
    
    return img


def _load_stats(params, data_type, inp_or_tar, stat_type='minmax'):
    """Load statistics for data normalization.
    
    Args:
        params: Parameter object containing paths and configurations
        data_type: String, one of ['sst', 'ocean', 'atmosphere']
        inp_or_tar: String, 'inp' for input or 'tar' for target
        stat_type: String, 'minmax' or 'zscore'
        
    Returns:
        tuple: (stat1, stat2, n_channels) where:
            - for minmax: (mins, maxs, n_channels)
            - for zscore: (means, stds, n_channels)
    """
    # Select channels based on input/target mode
    channels = getattr(params, f'{data_type}_in_channels') if inp_or_tar == 'inp' else \
              getattr(params, f'{data_type}_out_channels')
    
    if stat_type == 'minmax':
        stat1 = np.load(getattr(params, f'{data_type}_global_mins_path'))[:, channels]
        stat2 = np.load(getattr(params, f'{data_type}_global_maxs_path'))[:, channels]
    else:  # zscore
        stat1 = np.load(getattr(params, f'{data_type}_global_means_path'))[:, channels]
        stat2 = np.load(getattr(params, f'{data_type}_global_stds_path'))[:, channels]
    
    return stat1, stat2, len(channels)


def _normalize(data, stat1, stat2, method='minmax'):
    """Apply normalization to data.
    
    Args:
        data: Input data to normalize
        stat1: First statistic (mins for minmax, means for zscore)
        stat2: Second statistic (maxs for minmax, stds for zscore)
        method: String, 'minmax' or 'zscore'
    
    Returns:
        Normalized data
    """
    if method == 'minmax':
        return (data - stat1) / (stat2 - stat1)
    elif method == 'zscore':  # zscore
        return (data - stat1) / stat2
    else:
        raise ValueError(f"Invalid normalization method: {method}")


def _denormalize(data, means, stds, method='zscore'):
    """Apply denormalization to data.
    
    Args:
        data: Input data to denormalize
        means: Mean values for zscore denormalization
        stds: Standard deviation values for zscore denormalization
        method: String, currently only supports 'zscore'
    
    Returns:
        Denormalized data
    """
    if method == 'zscore':
        return data * stds + means
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    