import h5py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from scipy.ndimage import gaussian_filter
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import requests


class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool, use_weight: bool=True):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize
        self.use_weight = use_weight

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            if pol.min() == 0:
                value = 2*pol-1
            else:
                value = pol

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        if self.use_weight:
                            interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())
                        else:
                            interp_weights = value

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

def events_to_voxel_grid(voxel_grid, bin, x, y, p, t, device: str='cpu'):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32') # -1 1
    return voxel_grid.convert(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(pol),
        torch.from_numpy(t))

def read_voxel_hdf5(file_path,key='event_voxels'):
    with h5py.File(file_path, 'r') as f:
        voxel_data = f[key][:]
    # B, H, W = voxel_data.shape
    # voxel_data = voxel_data.reshape(3, 2, H, W).sum(axis=1)  # New shape: (B//2, H, W)
    return voxel_data

def resize_event_voxel(event_voxel, target_size, mode='bilinear'):
    """
    Adjust spatial resolution (H, W) of event data, preserving temporal dimension (C).
    
    Args:
        event_voxel (torch.Tensor): Event data, shape [C, H, W]
        target_size (tuple): Target resolution (target_h, target_w)
        mode (str): Interpolation mode, 'bilinear' or 'nearest'
    
    Returns:
        torch.Tensor: Resized data, shape [C, target_h, target_w]
    """
    C, H, W = event_voxel.shape
    target_h, target_w = target_size
    
    # Use interpolate to adjust spatial dimensions, keep C dimension unchanged
    event_voxel = F.interpolate(event_voxel.unsqueeze(0), size=(target_h, target_w), mode=mode, align_corners=False if mode == 'bilinear' else None)
    return event_voxel.squeeze(0)  # Remove temporarily added batch dimension

def crop_event(event_voxel, size, square_ok=False, crop=True, mode='bilinear'):
    """
    Crop or resize event data to align with crop_img.
    
    Args:
        event_voxel (torch.Tensor): Event data, shape [C, H, W]
        size (int): Target size (consistent with crop_img size parameter, e.g., 224 or 512)
        square_ok (bool): Whether to allow square output
        crop (bool): Whether to crop (True) or resize (False)
        mode (str): Interpolation mode, 'bilinear' or 'nearest'
    
    Returns:
        torch.Tensor: Cropped or resized data
    """
    C, H1, W1 = event_voxel.shape
    
    # Step 1: Resize to align with crop_img
    if size == 224:
        # Resize short edge to 224
        scale = size / min(H1, W1)
        target_h = round(H1 * scale)
        target_w = round(W1 * scale)
        event_voxel = resize_event_voxel(event_voxel, (target_h, target_w), mode=mode)
    else:
        # Resize long edge to 512
        scale = size / max(H1, W1)
        target_h = round(H1 * scale)
        target_w = round(W1 * scale)
        event_voxel = resize_event_voxel(event_voxel, (target_h, target_w), mode=mode)

    # Step 2: Crop or resize to target region
    C, H, W = event_voxel.shape
    cx, cy = W // 2, H // 2

    if size == 224:
        # Crop to 224x224
        half = min(cx, cy)
        left = cx - half
        top = cy - half
        right = cx + half
        bottom = cy + half
        event_voxel = event_voxel[:, top:bottom, left:right]
    else:
        # Crop to 512x384 or resize
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not square_ok and W == H:
            halfh = 3 * halfw // 4  # Adjust to 3:4 aspect ratio
        
        if crop:
            left = cx - halfw
            top = cy - halfh
            right = cx + halfw
            bottom = cy + halfh
            event_voxel = event_voxel[:, top:bottom, left:right]
        else:
            # Resize instead of crop
            target_size = (2 * halfw, 2 * halfh)
            event_voxel = resize_event_voxel(event_voxel, target_size, mode=mode)

    return event_voxel

def detect_harris_corners_and_gradients(image, patch_size=5, k=0.04, threshold=0.01):
    """
    Detect Harris corners and compute gradients
    """
    # 1. Compute image gradients
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # 2. Compute gradient products
    Ixx = Ix * Ix
    Ixy = Iy * Ix
    Iyy = Iy * Iy
    
    # 3. Smooth using Gaussian kernel
    Ixx = gaussian_filter(Ixx, sigma=1)
    Ixy = gaussian_filter(Ixy, sigma=1)
    Iyy = gaussian_filter(Iyy, sigma=1)
    
    # 4. Compute Harris response
    det = (Ixx * Iyy) - (Ixy * Ixy)
    trace = Ixx + Iyy
    harris_response = det - k * (trace ** 2)
    
    # 5. Non-maximum suppression
    corners = []
    patches = []
    gradients = []
    
    # Get local maxima
    local_max = cv2.dilate(harris_response, None)
    corner_mask = (harris_response == local_max) & (harris_response > threshold * harris_response.max())
    corner_points = np.where(corner_mask)
    
    # 6. Extract patches and gradients around corners
    half_patch = patch_size // 2
    for i in range(len(corner_points[0])):
        y, x = corner_points[0][i], corner_points[1][i]
        
        # Ensure patch does not exceed image boundaries
        if (y >= half_patch and y < image.shape[0] - half_patch and 
            x >= half_patch and x < image.shape[1] - half_patch):
            
            # Extract patch
            patch = image[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
            
            # Compute gradient at this point
            grad_x = Ix[y, x].mean()  # Use mean() to get scalar value
            grad_y = Iy[y, x].mean()  # Use mean() to get scalar value
            
            corners.append([x, y])
            patches.append(patch)
            gradients.append([grad_x, grad_y])
    
    return np.array(corners), np.array(patches), np.array(gradients)


# Global variables for caching SuperPoint model
superpoint_processor = None
superpoint_model = None

def get_superpoint_model(device='cuda'):
    """
    Get SuperPoint model, support GPU execution, ensure in inference mode
    """
    global superpoint_processor, superpoint_model
    
    if superpoint_processor is None or superpoint_model is None:
        superpoint_processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        superpoint_model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
        
        # Ensure model is in inference mode
        superpoint_model.eval()
        
        # Move model to specified device
        if device == 'cuda' and torch.cuda.is_available():
            superpoint_model = superpoint_model.to('cuda')
            print("SuperPoint model moved to GPU (inference mode)")
        else:
            print("SuperPoint model running on CPU (inference mode)")
    
    return superpoint_processor, superpoint_model
def generate_superpoint_points(image, patch_size=5, device='cuda'):
    """
    Generate keypoints using SuperPoint, support GPU execution, ensure no gradient computation
    """
    # 1. Compute image gradients
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Get SuperPoint model
    processor, model = get_superpoint_model(device)
    
    # Ensure model is in inference mode and does not participate in gradient computation
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # 5. Non-maximum suppression
    corners = []
    patches = []
    gradients = []
    
    # Prepare input data
    inputs = processor(image, return_tensors="pt", do_rescale=False, use_fast=True)
    
    # Move input to GPU
    if device == 'cuda' and torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Run inference - ensure no gradient computation
    with torch.inference_mode():
        outputs = model(**inputs)
    
    corner_points = outputs['keypoints'][0]   # n,2
    
    # If result is on GPU, move to CPU
    if corner_points.device.type == 'cuda':
        corner_points = corner_points.cpu().numpy()
    else:
        corner_points = corner_points.numpy()
    
    # SuperPoint outputs normalized coordinates (0-1), need to convert to pixel coordinates
    height, width = image.shape[:2]
    corner_points = corner_points * np.array([width, height])  # Convert to pixel coordinates
    
    # 6. Extract patches and gradients around corners
    half_patch = patch_size // 2
    for i in range(len(corner_points)):
        x, y = corner_points[i]
        
        # Convert to integer coordinates
        x, y = int(x), int(y)
        
        # Ensure patch does not exceed image boundaries
        if (y >= half_patch and y < image.shape[0] - half_patch and 
            x >= half_patch and x < image.shape[1] - half_patch):
            
            # Extract patch
            patch = image[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
            
            # Compute gradient at this point
            grad_x = Ix[y, x].mean()  # Use mean() to get scalar value
            grad_y = Iy[y, x].mean()  # Use mean() to get scalar value
            
            corners.append([x, y])
            patches.append(patch)
            gradients.append([grad_x, grad_y])
    
    return np.array(corners), np.array(patches), np.array(gradients)

def generate_random_points(image, patch_size=5, num_points=500):
    """
    Generate random points for comparison experiments
    """
    # 1. Compute image gradients
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    
    # 5. Non-maximum suppression
    corners = []
    patches = []
    gradients = []
    
    # Randomly generate num_points points
    height, width = image.shape[:2]
    corner_points = np.random.randint(0, [width, height], (num_points, 2))
    
    # 6. Extract patches and gradients around corners
    half_patch = patch_size // 2
    for i in range(len(corner_points)):
        x, y = corner_points[i]
        
        # Ensure patch does not exceed image boundaries
        if (y >= half_patch and y < image.shape[0] - half_patch and 
            x >= half_patch and x < image.shape[1] - half_patch):
            
            # Extract patch
            patch = image[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
            
            # Compute gradient at this point
            grad_x = Ix[y, x].mean()  # Use mean() to get scalar value
            grad_y = Iy[y, x].mean()  # Use mean() to get scalar value
            
            corners.append([x, y])
            patches.append(patch)
            gradients.append([grad_x, grad_y])
    
    return np.array(corners), np.array(patches), np.array(gradients)

def detect_harris_corners_fast(image, patch_size=5, k=0.04, threshold=0.01, max_corners=1000):
    """
    Use OpenCV built-in Harris corner detection, more efficient - further optimized version
    """
    # Ensure image format is correct
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.dtype != np.uint8:
        # If float, convert to 0-255 range
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Ensure image is single channel
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV Harris corner detection - optimized parameters
    corners = cv2.goodFeaturesToTrack(
        image, 
        maxCorners=max_corners,
        qualityLevel=threshold,
        minDistance=patch_size//2,  # Reduce minimum distance to get more corners
        blockSize=3,  # Use smaller block size for speed
        useHarrisDetector=True,
        k=k
    )
    
    if corners is None:
        return np.array([]), np.array([]), np.array([])
    
    corners = corners.reshape(-1, 2)
    
    # Batch compute gradients - optimized version
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    # Batch extract patches and gradients - vectorized operations
    half_patch = patch_size // 2
    height, width = image.shape
    
    # Pre-filter valid corners
    valid_mask = (
        (corners[:, 1] >= half_patch) & (corners[:, 1] < height - half_patch) &
        (corners[:, 0] >= half_patch) & (corners[:, 0] < width - half_patch)
    )
    
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
    
    valid_corners = corners[valid_mask]
    
    # Batch extract patches
    y_coords = valid_corners[:, 1].astype(int)
    x_coords = valid_corners[:, 0].astype(int)
    
    # Batch extract patches - use numpy advanced indexing
    patches = []
    for y, x in zip(y_coords, x_coords):
        patch = image[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
        patches.append(patch)
    
    # Batch extract gradients
    gradients = np.column_stack([
        Ix[y_coords, x_coords],
        Iy[y_coords, x_coords]
    ])
    
    return valid_corners, np.array(patches), gradients

def visualize_corners_and_gradients(image, corners, gradients):
    """
    Visualize corners and gradients
    """
    plt.figure(figsize=(12, 4))
    
    # Display original image
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    # Display corners
    plt.subplot(132)
    plt.imshow(image, cmap='gray')
    plt.scatter(corners[:, 0], corners[:, 1], c='r', s=20)
    plt.title('Detected Corners')
    
    # Display gradients
    plt.subplot(133)
    plt.imshow(image, cmap='gray')
    
    # Ensure gradient and corner counts match
    if len(corners) > 0 and len(gradients) > 0:
        # Normalize gradients for better visualization
        norm = np.sqrt(gradients[:, 0]**2 + gradients[:, 1]**2)
        norm[norm == 0] = 1  # Avoid division by zero
        normalized_gradients = gradients / norm[:, np.newaxis]
        
        # Use quiver to display gradients
        plt.quiver(corners[:, 0], corners[:, 1], 
                  normalized_gradients[:, 0], normalized_gradients[:, 1],
                  color='r', scale=50)
    plt.title('Gradients at Corners')
    
    plt.tight_layout()
    plt.show()

def compute_gradient_flow_dot_product(corners, gradients, ego_flow):
    """
    Compute dot product of gradients and optical flow at each corner, preserve gradient information for loss calculation
    
    Args:
        corners: Corner coordinates [N, 2]
        gradients: Gradient vectors [N, 2]
        ego_flow: Optical flow field [2, H, W] (CUDA tensor)
    
    Returns:
        dot_products: Dot product results [N] (preserve gradient information)
    """
    # Check if input is empty
    if corners is None or len(corners) == 0:
        # Return empty tensor on same device as ego_flow
        return torch.empty(0, device=ego_flow.device, dtype=ego_flow.dtype)
    
    dot_products = []
    
    for i in range(len(corners)):
        x, y = corners[i].astype(int)
        gx, gy = gradients[i]
        
        # Get optical flow at this point
        flow_x = ego_flow[0, y, x]  # Optical flow in x direction
        flow_y = ego_flow[1, y, x]  # Optical flow in y direction
        
        # Compute dot product (preserve gradient information)
        dot_product = gx * flow_x + gy * flow_y
        dot_products.append(dot_product)
    
    # Check if there are valid dot product results
    if len(dot_products) == 0:
        return torch.empty(0, device=ego_flow.device, dtype=ego_flow.dtype)
    
    return torch.stack(dot_products)  # Return tensor instead of numpy array

def compute_gradient_flow_dot_product_fast(corners, gradients, ego_flow):
    """
    Compute dot product of gradients and optical flow at each corner, preserve gradient information for loss calculation - optimized version
    
    Args:
        corners: Corner coordinates [N, 2]
        gradients: Gradient vectors [N, 2]
        ego_flow: Optical flow field [2, H, W] (CUDA tensor)
    
    Returns:
        dot_products: Dot product results [N] (preserve gradient information)
    """
    # Check if input is empty
    if corners is None or len(corners) == 0:
        # Return empty tensor on same device as ego_flow
        return torch.empty(0, device=ego_flow.device, dtype=ego_flow.dtype)
    
    # Convert to tensor and ensure on correct device
    if not isinstance(corners, torch.Tensor):
        corners = torch.from_numpy(corners).to(ego_flow.device)
    if not isinstance(gradients, torch.Tensor):
        gradients = torch.from_numpy(gradients).to(ego_flow.device)
    
    # Check again if converted tensor is empty
    if corners.numel() == 0:
        return torch.empty(0, device=ego_flow.device, dtype=ego_flow.dtype)
    
    # Batch indexing operation, avoid loops
    x = corners[:, 0].long()
    y = corners[:, 1].long()
    
    # Batch get optical flow values [N]
    flow_x = ego_flow[0, y, x]
    flow_y = ego_flow[1, y, x]
    
    # Batch compute dot products [N]
    dot_products = gradients[:, 0] * flow_x + gradients[:, 1] * flow_y
    
    return dot_products

def batch_index_tensor(tensor, indices, device=None):
    """
    Efficient batch indexing operation - optimized version
    
    Args:
        tensor: Tensor to index [C, H, W] or [H, W]
        indices: Index coordinates [N, 2] (x, y)
        device: Target device
    
    Returns:
        indexed_values: Indexing results [N] or [C, N]
    """
    # Check if input is empty
    if indices is None or len(indices) == 0:
        # Return empty tensor on same device as input tensor
        if tensor.dim() == 3:  # [C, H, W]
            return torch.empty(tensor.shape[0], 0, device=tensor.device, dtype=tensor.dtype)
        else:  # [H, W]
            return torch.empty(0, device=tensor.device, dtype=tensor.dtype)
    
    # Process input - optimized version
    if not isinstance(indices, torch.Tensor):
        indices = torch.from_numpy(indices)
    
    # Ensure indices and tensor are on same device - simplified version
    if device is not None:
        indices = indices.to(device)
        tensor = tensor.to(device)
    else:
        indices = indices.to(tensor.device)
    
    # Check again if converted indices are empty
    if indices.numel() == 0:
        if tensor.dim() == 3:  # [C, H, W]
            return torch.empty(tensor.shape[0], 0, device=tensor.device, dtype=tensor.dtype)
        else:  # [H, W]
            return torch.empty(0, device=tensor.device, dtype=tensor.dtype)
    
    # Ensure indices are within valid range - optimized version
    H, W = tensor.shape[-2:]
    max_x = W - 1
    max_y = H - 1
    
    # Batch clamp coordinates - fixed version
    indices[:, 0] = torch.clamp(indices[:, 0], min=0, max=max_x)
    indices[:, 1] = torch.clamp(indices[:, 1], min=0, max=max_y)
    
    # Ensure index data type is correct - fixed version
    indices = indices.long()  # Convert to long type
    
    # Batch indexing - optimized version
    y, x = indices[:, 1], indices[:, 0]
    
    # Direct indexing, avoid try-except overhead
    if tensor.dim() == 3:  # [C, H, W]
        return tensor[:, y, x]  # [C, N]
    else:  # [H, W]
        return tensor[y, x]  # [N]

def visualize_gradient_flow_dot_product(image, corners, gradients, ego_flow, patch_flow, dot_products, event_repr_at_corners=None, save_dir='visualization'):
    """
    Visualize dot product results of gradients and optical flow and save to local
    
    Args:
        image: Input image
        corners: Corner coordinates [N, 2]
        gradients: Gradient vectors [N, 2]
        ego_flow: Optical flow field [2, H, W]
        patch_flow: Optical flow at corners [2, N]
        dot_products: Dot product results [N]
        event_repr_at_corners: Event representation at corners [N], optional
        save_dir: Save directory
    """
    # For visualization, we need to detach gradients and convert to numpy
    image = image.detach().cpu().numpy() if torch.is_tensor(image) else image
    corners = corners.detach().cpu().numpy() if torch.is_tensor(corners) else corners
    gradients = gradients.detach().cpu().numpy() if torch.is_tensor(gradients) else gradients
    ego_flow = ego_flow.detach().cpu().numpy() if torch.is_tensor(ego_flow) else ego_flow
    patch_flow = patch_flow.detach().cpu().numpy() if torch.is_tensor(patch_flow) else patch_flow
    dot_products_np = dot_products.detach().cpu().numpy() if torch.is_tensor(dot_products) else dot_products
    if event_repr_at_corners is not None:
        event_repr_np = event_repr_at_corners.detach().cpu().numpy() if torch.is_tensor(event_repr_at_corners) else event_repr_at_corners
    
    # Ensure patch_flow dimension is correct [2, N] -> [N, 2]
    if patch_flow.shape[0] == 2:
        patch_flow = patch_flow.T
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure and GridSpec
    if event_repr_at_corners is not None:
        fig = plt.figure(figsize=(30, 5))
        gs = fig.add_gridspec(1, 5)
    else:
        fig = plt.figure(figsize=(24, 5))
        gs = fig.add_gridspec(1, 4)
    
    # 1. Display original image and corners
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.scatter(corners[:, 0], corners[:, 1], c='r', s=20)
    ax1.set_title('Corner Positions')
    
    # 2. Display gradients and optical flow
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image, cmap='gray')
    # Normalize gradients
    norm = np.sqrt(gradients[:, 0]**2 + gradients[:, 1]**2)
    norm[norm == 0] = 1
    normalized_gradients = gradients / norm[:, np.newaxis]
    
    # Draw gradients
    ax2.quiver(corners[:, 0], corners[:, 1], 
              normalized_gradients[:, 0], normalized_gradients[:, 1],
              color='r', scale=50, label='Gradient')
    
    # Draw optical flow
    for i in range(len(corners)):
        x, y = corners[i].astype(int)
        flow_x = ego_flow[0, y, x]  # Optical flow in x direction
        flow_y = ego_flow[1, y, x]  # Optical flow in y direction
        ax2.arrow(x, y, flow_x, flow_y, 
                 color='b', alpha=0.5, 
                 head_width=2, head_length=2,
                 label='Flow' if i==0 else "")
    
    ax2.set_title('Gradients and Flow')
    ax2.legend()
    
    # 3. Display dot product results
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image, cmap='gray')
    scatter = ax3.scatter(corners[:, 0], corners[:, 1], 
                         c=dot_products_np, cmap='coolwarm', s=50)
    ax3.set_title('Gradient-Flow Dot Product')
    
    # Add colorbar and adjust position
    cbar = plt.colorbar(scatter, ax=ax3, pad=0.1)
    cbar.set_label('Dot Product')
    
    # 4. Display patch flow
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(image, cmap='gray')
    # Normalize patch flow
    norm = np.sqrt(patch_flow[:, 0]**2 + patch_flow[:, 1]**2)
    norm[norm == 0] = 1
    normalized_patch_flow = patch_flow / norm[:, np.newaxis]
    
    # Draw patch flow
    ax4.quiver(corners[:, 0], corners[:, 1], 
              normalized_patch_flow[:, 0], normalized_patch_flow[:, 1],
              color='g', scale=50)
    ax4.set_title('Patch Flow')
    
    # 5. If event representation is provided, display it
    if event_repr_at_corners is not None:
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(image, cmap='gray')
        scatter = ax5.scatter(corners[:, 0], corners[:, 1], 
                            c=event_repr_np, cmap='coolwarm', s=50)
        ax5.set_title('Event Representation')
        
        # Add colorbar and adjust position
        cbar = plt.colorbar(scatter, ax=ax5, pad=0.1)
        cbar.set_label('Event Value')
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.3, right=0.95)
    
    # Save image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'gradient_flow_dot_product_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save dot product data
    data_save_path = os.path.join(save_dir, f'dot_products_{timestamp}.npy')
    save_dict = {
        'corners': corners,
        'gradients': gradients,
        'dot_products': dot_products_np,
        'ego_flow': ego_flow,
        'patch_flow': patch_flow
    }
    if event_repr_at_corners is not None:
        save_dict['event_repr'] = event_repr_np
    np.save(data_save_path, save_dict)
    
    # Print statistics
    print(f"Dot product range: [{dot_products_np.min():.2f}, {dot_products_np.max():.2f}]")
    print(f"Mean dot product: {dot_products_np.mean():.2f}")
    print(f"Dot product std: {dot_products_np.std():.2f}")
    if event_repr_at_corners is not None:
        print(f"Event representation range: [{event_repr_np.min():.2f}, {event_repr_np.max():.2f}]")
        print(f"Mean event representation: {event_repr_np.mean():.2f}")
        print(f"Event representation std: {event_repr_np.std():.2f}")
    print(f"Image saved to: {save_path}")
    print(f"Data saved to: {data_save_path}")

def normalized_l2_loss(gt, pred, eps=1e-8):
    """
    Normalized L2 distance loss
    gt: torch.Tensor, shape [N] or [B, N]
    pred: torch.Tensor, shape [N] or [B, N]
    """
    # Check if input is empty
    if gt.numel() == 0 or pred.numel() == 0:
        # Return zero loss on same device as input
        return torch.tensor(0.0, device=gt.device, requires_grad=True)
    
    # Ensure float type
    gt = gt.float()
    pred = pred.float()
    
    # Normalize
    gt_norm = gt / (torch.norm(gt, p=2, dim=-1, keepdim=True) + eps)
    pred_norm = pred / (torch.norm(pred, p=2, dim=-1, keepdim=True) + eps)
    
    # Squared L2 distance
    loss = torch.sum((gt_norm - pred_norm) ** 2, dim=-1)
    return loss.mean()  # If batch exists, take mean; otherwise scalar

def normalized_l2_loss_fast(gt, pred, eps=1e-8):
    """
    Normalized L2 distance loss - optimized version
    gt: torch.Tensor, shape [N] or [B, N]
    pred: torch.Tensor, shape [N] or [B, N]
    """
    # Check if input is empty
    if gt.numel() == 0 or pred.numel() == 0:
        # Return zero loss on same device as input
        return torch.tensor(0.0, device=gt.device, requires_grad=True)
    
    # Ensure device consistency - optimized version
    if gt.device != pred.device:
        pred = pred.to(gt.device)
    
    # Ensure float type and merge operations
    gt = gt.float()
    pred = pred.float()
    
    # Optimize normalization calculation - avoid repeated computation
    gt_norm = torch.norm(gt, p=2, dim=-1, keepdim=True)
    pred_norm = torch.norm(pred, p=2, dim=-1, keepdim=True)
    
    # Avoid division by zero
    gt_normalized = gt / (gt_norm + eps)
    pred_normalized = pred / (pred_norm + eps)
    
    # Use more efficient L2 distance calculation
    diff = gt_normalized - pred_normalized
    loss = torch.sum(diff * diff, dim=-1)  # Use square instead of **2
    
    return loss.mean()  # If batch exists, take mean; otherwise scalar

def visualize_flow(flow, image=None, save_dir='visualization', title='Flow Visualization'):
    """
    Visualize optical flow field
    
    Args:
        flow: Optical flow field [2, H, W]
        image: Original image, optional
        save_dir: Save directory
        title: Image title
    """
    # Convert to numpy array
    flow = flow.detach().cpu().numpy() if torch.is_tensor(flow) else flow
    if image is not None:
        image = image.detach().cpu().numpy() if torch.is_tensor(image) else image
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure and subplots
    fig = plt.figure(figsize=(20, 5))
    if image is not None:
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
    else:
        ax2 = fig.add_subplot(121)
        ax3 = fig.add_subplot(122)
    
    # 1. Display original image (if available)
    if image is not None:
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image')
    
    # 2. Display optical flow field (arrow representation)
    if image is not None:
        ax2.imshow(image, cmap='gray')
    
    # Create grid points
    h, w = flow.shape[1:]
    y, x = np.mgrid[0:h:20, 0:w:20]  # Sample one point every 20 pixels
    
    # Get optical flow values at sample points
    flow_x = flow[0, y.flatten(), x.flatten()]  # Flatten coordinates
    flow_y = flow[1, y.flatten(), x.flatten()]
    
    # Compute optical flow magnitude for color mapping
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    
    # Draw optical flow field
    quiver = ax2.quiver(x.flatten(), y.flatten(), flow_x, flow_y, magnitude,
              cmap='viridis', scale=50,
              width=0.003, headwidth=3)
    
    # Add colorbar
    cbar = fig.colorbar(quiver, ax=ax2, pad=0.1)
    cbar.set_label('Flow Magnitude')
    ax2.set_title('Flow Field (Arrows)')
    
    # 3. Display optical flow field (color mapping)
    # Compute optical flow magnitude and direction
    magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
    angle = np.arctan2(flow[1], flow[0])
    
    # Normalize angle to [0, 1] range
    angle = (angle + np.pi) / (2 * np.pi)
    
    # Create HSV color mapping
    hsv = np.zeros((h, w, 3))
    hsv[..., 0] = angle  # Hue: represents direction
    hsv[..., 1] = 1.0    # Saturation: set to maximum
    hsv[..., 2] = np.clip(magnitude / magnitude.max(), 0, 1)  # Brightness: represents magnitude
    
    # Convert to RGB
    rgb = plt.cm.hsv(hsv[..., 0])
    rgb[..., 3] = hsv[..., 2]  # Use alpha channel to represent magnitude
    
    # Display color-mapped optical flow field
    ax3.imshow(rgb)
    ax3.set_title('Flow Field (Color Map)')
    
    # Add direction legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', label='Right'),
        plt.Line2D([0], [0], color='green', label='Up'),
        plt.Line2D([0], [0], color='blue', label='Left'),
        plt.Line2D([0], [0], color='yellow', label='Down')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.3, right=0.95)
    
    # Save image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'flow_visualization_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Optical flow visualization saved to: {save_path}")
    print(f"Optical flow magnitude range: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
    print(f"Mean optical flow magnitude: {magnitude.mean():.2f}")
    print(f"Optical flow magnitude std: {magnitude.std():.2f}")