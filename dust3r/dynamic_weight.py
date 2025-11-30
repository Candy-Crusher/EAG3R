#!/usr/bin/env python3
"""
Dynamic weight adjustment function
Dynamically adjust event_loss_weight based on SNR and corner count
"""

import numpy as np
import torch

def calculate_dynamic_event_loss_weight(snr_mean, corner_count, base_weight=0.01):
    """
    Calculate dynamic event_loss_weight based on SNR and corner count
    
    Args:
        snr_mean: Mean SNR value
        corner_count: Number of corners
        base_weight: Base weight (default 0.01)
    
    Returns:
        dynamic_weight: Dynamically adjusted weight
    """
    
    # Dynamic weight strategy based on analysis results
    if snr_mean > 0.62:  # High SNR sequence
        if corner_count < 150:  # Few corners
            # High SNR + Few corners: event_loss works best
            dynamic_weight = base_weight * 1.5  # 0.015
        elif corner_count < 200:  # Medium corners
            # High SNR + Medium corners: Good effect
            dynamic_weight = base_weight * 1.2  # 0.012
        else:  # Many corners
            # High SNR + Many corners: Medium effect
            dynamic_weight = base_weight * 0.8  # 0.008
            
    elif snr_mean > 0.58:  # Medium SNR sequence
        if corner_count < 150:  # Few corners
            # Medium SNR + Few corners: Good effect
            dynamic_weight = base_weight * 1.1  # 0.011
        elif corner_count < 200:  # Medium corners
            # Medium SNR + Medium corners: Standard weight
            dynamic_weight = base_weight  # 0.01
        else:  # Many corners
            # Medium SNR + Many corners: Average effect
            dynamic_weight = base_weight * 0.7  # 0.007
            
    else:  # Low SNR sequence
        if corner_count < 150:  # Few corners
            # Low SNR + Few corners: Average effect
            dynamic_weight = base_weight * 0.5  # 0.005
        elif corner_count < 200:  # Medium corners
            # Low SNR + Medium corners: Poor effect
            dynamic_weight = base_weight * 0.3  # 0.003
        else:  # Many corners
            # Low SNR + Many corners: event_loss works worst, recommend not using
            dynamic_weight = base_weight * 0.1  # 0.001
    
    # Ensure weight is in reasonable range
    dynamic_weight = max(0.0, min(dynamic_weight, 0.02))
    
    return dynamic_weight

def get_snr_and_corners_from_model(model, imgs):
    """
    Get SNR and corner information from model
    
    Args:
        model: Model instance
        imgs: Image data
    
    Returns:
        snr_mean: Mean SNR value
        corner_count: Number of corners
    """
    try:
        import cv2
        
        # Get original model (may be wrapped by DistributedDataParallel)
        original_model = model.module if hasattr(model, 'module') else model
        
        # Get SNR map
        snr_mean = 0.0
        if hasattr(original_model, '_last_snr_map') and original_model._last_snr_map is not None:
            snr_map = original_model._last_snr_map
            if isinstance(snr_map, torch.Tensor):
                snr_map_np = snr_map.detach().cpu().numpy()
                snr_mean = float(np.mean(snr_map_np))
        
        # Compute corner count (using first image)
        corner_count = 0
        if len(imgs) > 0:
            img = imgs[0]['img']
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img.squeeze(0)
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img
            
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            # Use Harris corner detection
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            corner_count = int(np.sum(corners > 0.01 * corners.max()))
        
        return snr_mean, corner_count
        
    except Exception as e:
        print(f"Error getting SNR and corners: {e}")
        return 0.0, 0

def log_dynamic_weight_info(seq_id, snr_mean, corner_count, base_weight, dynamic_weight):
    """
    Log dynamic weight adjustment information
    
    Args:
        seq_id: Sequence ID
        snr_mean: Mean SNR value
        corner_count: Number of corners
        base_weight: Base weight
        dynamic_weight: Dynamic weight
    """
    print(f"[Dynamic Weight] Seq {seq_id}: SNR={snr_mean:.4f}, Corners={corner_count}, "
          f"Base={base_weight:.4f} -> Dynamic={dynamic_weight:.4f} "
          f"(Factor={dynamic_weight/base_weight:.2f})")

if __name__ == "__main__":
    # Test dynamic weight function
    print("=== Dynamic Weight Test ===")
    
    test_cases = [
        (0.65, 120, "High SNR + Few Corners"),
        (0.65, 180, "High SNR + Medium Corners"),
        (0.65, 220, "High SNR + Many Corners"),
        (0.60, 120, "Medium SNR + Few Corners"),
        (0.60, 180, "Medium SNR + Medium Corners"),
        (0.60, 220, "Medium SNR + Many Corners"),
        (0.55, 120, "Low SNR + Few Corners"),
        (0.55, 180, "Low SNR + Medium Corners"),
        (0.55, 220, "Low SNR + Many Corners"),
    ]
    
    for snr, corners, desc in test_cases:
        weight = calculate_dynamic_event_loss_weight(snr, corners)
        print(f"{desc}: SNR={snr:.2f}, Corners={corners:3d} -> Weight={weight:.4f}")
