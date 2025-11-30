#!/usr/bin/env python3
"""
SNR-weighted event loss computation
Directly use snr_map as weights in event loss calculation
"""

import torch
import torch.nn.functional as F
import numpy as np

def compute_snr_weighted_event_loss(ev, dp, snr_map, corners, snr_weight_strategy="1-snr", snr_weight_type="pixel", eps=1e-8):
    """
    Weight event loss using SNR map
    
    Args:
        ev: Event loss component
        dp: Depth/flow loss component
        snr_map: SNR map [H, W]
        corners: Corner coordinates [N, 2]
        snr_weight_strategy: Weighting strategy ('snr' or '1-snr')
        snr_weight_type: Weighting type ('pixel' or 'region')
        eps: Small value to avoid division by zero
    
    Returns:
        weighted_losses: Weighted loss value
    """
    if ev.numel() == 0 or dp.numel() == 0:
        return torch.tensor(0.0, device=ev.device, requires_grad=True)
    
    # Ensure device consistency
    if ev.device != dp.device:
        dp = dp.to(ev.device)
    if snr_map.device != ev.device:
        snr_map = snr_map.to(ev.device)
    if corners.device != ev.device:
        corners = corners.to(ev.device)
    
    # Get SNR values at corner positions
    # corners: [N, 2] -> (x, y)
    # snr_map: [H, W]
    corner_y = corners[:, 1].long().clamp(0, snr_map.shape[0] - 1)
    corner_x = corners[:, 0].long().clamp(0, snr_map.shape[1] - 1)
    
    corner_snr = snr_map[corner_y, corner_x]  # [N]
    
    # Compute SNR weights - consider two strategies
    # Strategy 1: Higher weight for high SNR regions (assume high SNR is more reliable)
    # Strategy 2: Higher weight for low SNR regions (assume low SNR needs more attention)

    if snr_weight_strategy == "1-snr":
        # Use strategy 2: Higher weight for low SNR regions (1-snr)
        # Low SNR regions usually have more noise and need more event loss constraint
        snr_weights = 1.0 - torch.sigmoid(corner_snr)  # [N] Higher weight for low SNR regions
    elif snr_weight_strategy == "snr":  # Use strategy 1: Higher weight for high SNR regions (snr)
        snr_weights = torch.sigmoid(corner_snr)  # [N] Higher weight for high SNR regions
    else:
        raise ValueError(f"Invalid SNR weight strategy: {snr_weight_strategy}")
    
    # Compute per-pixel loss (true pixel-level)
    ev_norm = torch.norm(ev, p=2)
    dp_norm = torch.norm(dp, p=2)
    
    ev_normalized = ev / (ev_norm + eps)
    dp_normalized = dp / (dp_norm + eps)
    
    
    if snr_weight_type == "pixel":
        # Pixel-level weighting: each pixel uses corresponding SNR weight
        pixel_losses = (ev_normalized - dp_normalized) ** 2  # [N] Per-pixel loss
        weighted_losses = (pixel_losses * snr_weights).mean()  # Average after pixel-level weighting
    elif snr_weight_type == "region":
        # Region-level weighting: each region uses corresponding SNR weight
        pixel_losses = torch.sum((ev_normalized - dp_normalized) ** 2)  # Scalar
        weighted_losses = pixel_losses * snr_weights.mean()  # Use average SNR weight
    else:
        raise ValueError(f"Invalid SNR weight type: {snr_weight_type}")
    
    return weighted_losses

def compute_adaptive_snr_weight(ev, dp, snr_map, corners, base_weight=0.01):
    """
    Adaptive weight adjustment based on SNR map
    
    Args:
        ev, dp: Event loss components
        snr_map: SNR map [H, W]
        corners: Corner coordinates [N, 2]
        base_weight: Base weight
    
    Returns:
        adaptive_weight: Adaptive weight
    """
    if snr_map.numel() == 0:
        return base_weight
    
    # Compute SNR statistics
    snr_mean = snr_map.mean()
    snr_std = snr_map.std()
    snr_quality = snr_mean / (snr_std + 1e-8)
    
    # Adjust weight based on quality
    if snr_quality > 2.0:  # High quality
        adaptive_weight = base_weight * 1.5
    elif snr_quality > 1.0:  # Medium quality
        adaptive_weight = base_weight * 1.0
    else:  # Low quality
        adaptive_weight = base_weight * 0.5
    
    return adaptive_weight

def compute_region_snr_weight(ev, dp, snr_map, corners, region_size=32):
    """
    Region-based SNR weight computation
    
    Args:
        ev, dp: Event loss components
        snr_map: SNR map [H, W]
        corners: Corner coordinates [N, 2]
        region_size: Region size
    
    Returns:
        region_weighted_loss: Region-weighted loss
    """
    if ev.numel() == 0 or dp.numel() == 0:
        return torch.tensor(0.0, device=ev.device, requires_grad=True)
    
    H, W = snr_map.shape
    regions_h = H // region_size
    regions_w = W // region_size
    
    weighted_losses = []
    
    for i in range(regions_h):
        for j in range(regions_w):
            # Get region boundaries
            y_start, y_end = i * region_size, (i + 1) * region_size
            x_start, x_end = j * region_size, (j + 1) * region_size
            
            # Get corners in this region
            region_mask = ((corners[:, 0] >= x_start) & (corners[:, 0] < x_end) & 
                          (corners[:, 1] >= y_start) & (corners[:, 1] < y_end))
            
            if region_mask.sum() > 0:
                # Compute region SNR weight
                region_snr = snr_map[y_start:y_end, x_start:x_end].mean()
                region_weight = torch.sigmoid(region_snr)
                
                # Get loss for this region
                region_ev = ev[region_mask]
                region_dp = dp[region_mask]
                
                if region_ev.numel() > 0:
                    region_loss = compute_snr_weighted_event_loss(
                        region_ev, region_dp, 
                        snr_map[y_start:y_end, x_start:x_end], 
                        corners[region_mask]
                    )
                    weighted_losses.append(region_weight * region_loss)
    
    if len(weighted_losses) > 0:
        return torch.stack(weighted_losses).mean()
    else:
        return torch.tensor(0.0, device=ev.device, requires_grad=True)

# Test function
if __name__ == "__main__":
    print("=== SNR-weighted Event Loss Test ===")
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate data
    N = 100
    H, W = 64, 64
    
    ev = torch.randn(N, device=device)
    dp = torch.randn(N, device=device)
    snr_map = torch.rand(H, W, device=device) * 2.0  # SNR range [0, 2]
    corners = torch.randint(0, min(H, W), (N, 2), device=device)
    
    # Test pixel-level SNR weighting
    loss1 = compute_snr_weighted_event_loss(ev, dp, snr_map, corners)
    print(f"Pixel-level SNR-weighted loss: {loss1.item():.6f}")
    
    # Test adaptive weight
    adaptive_weight = compute_adaptive_snr_weight(ev, dp, snr_map, corners)
    print(f"Adaptive weight: {adaptive_weight:.6f}")
    
    # Test region-level weighting
    loss2 = compute_region_snr_weight(ev, dp, snr_map, corners)
    print(f"Region-level SNR-weighted loss: {loss2.item():.6f}")
