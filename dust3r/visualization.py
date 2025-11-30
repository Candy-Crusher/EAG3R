import numpy as np
import matplotlib.pyplot as plt
import math

def visualize_feature(image_feature, save_path="visualization.png", true_shape=None, dim=100):
    """
    Visualize a single image feature tensor, display specified dimension and reshape to [H, W].

    Args:
        image_feature: Input feature tensor, shape [batch, seq_len, embed_dim]
        save_path: Path to save image, default "visualization.png"
        true_shape: True shape, shape [batch, 2], e.g., [[height, width]]
        dim: Dimension index to visualize, default 100
    """
    # Select first batch data
    batch_idx = 0
    if true_shape is None:
        raise ValueError("true_shape must be provided for reshaping")
    scale = math.sqrt(true_shape[batch_idx][0].item()*true_shape[batch_idx][1].item()/image_feature.shape[1])
    scale = int(scale)
    print(f"scale: {scale}")
    resize_shape = (true_shape[batch_idx][0].item() // scale, true_shape[batch_idx][1].item() // scale)
    print(f"resize_shape: {resize_shape}")
    H, W = resize_shape
    seq_len = H * W

    # Extract first batch data
    image_feature = image_feature[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]

    # Verify seq_len matches
    if image_feature.shape[0] != seq_len:
        raise ValueError(f"seq_len ({image_feature.shape[0]}) does not match expected H*W ({seq_len})")

    # Ensure dim is within embed_dim range
    embed_dim = image_feature.shape[1]
    if dim >= embed_dim:
        raise ValueError(f"Dimension index {dim} exceeds embed_dim {embed_dim}")

    # Reshape to [H, W], take specified dimension
    feature_vis = image_feature[:, dim].reshape(H, W)  # [H, W]

    # Normalize to [0, 1] for display (normalization commented out, keep as original code)
    def normalize(tensor):
        # tensor = tensor - tensor.min()
        # tensor = tensor / (tensor.max() + 1e-8)
        return tensor

    feature_vis = normalize(feature_vis)

    # Create single image
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Visualize image_feature (heatmap)
    im = ax.imshow(feature_vis, cmap='viridis')
    ax.set_title(f'Image Feature (dim {dim})')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im, ax=ax)

    # Save image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

def visualize_tensors_snr(x, event_feat, snr_map, output_x, save_path="visualization.png", true_shape=None):
    """
    Visualize tensors x, event_feat, snr_map and output_x, display first dimension only, reshape to [H, W].

    Args:
        x: Initial input tensor, shape [batch, seq_len, embed_dim]
        event_feat: Event feature tensor, same shape as x
        snr_map: Signal-to-noise ratio map, shape [batch, seq_len, 1]
        output_x: Final output tensor, same shape as x
        save_path: Path to save image, default "visualization.png"
        true_shape: True shape, shape [batch, 2], e.g., [[height, width]]
    """
    # Select first batch data
    batch_idx = 0
    if true_shape is None:
        raise ValueError("true_shape must be provided for reshaping")
    resize_shape = (true_shape[batch_idx][0].item() // 16, true_shape[batch_idx][1].item() // 16)
    H, W = resize_shape
    seq_len = H * W

    # Extract first batch data
    x = x[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    event_feat = event_feat[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    output_x = output_x[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    if snr_map is not None:
        snr_map = snr_map[batch_idx, :, 0].detach().cpu().numpy()  # [seq_len]

    # Verify seq_len matches
    if x.shape[0] != seq_len:
        raise ValueError(f"seq_len ({x.shape[0]}) does not match expected H*W ({seq_len})")

    # Reshape to [H, W], take first dimension
    x_vis = x[:, 100].reshape(H, W)  # [H, W]
    event_feat_vis = event_feat[:, 100].reshape(H, W)  # [H, W]
    output_x_vis = output_x[:, 100].reshape(H, W)  # [H, W]
    if snr_map is not None:
        snr_map = snr_map.reshape(H, W)  # [H, W]

    # Normalize to [0, 1] for display
    def normalize(tensor):
        # tensor = tensor - tensor.min()
        # tensor = tensor / (tensor.max() + 1e-8)
        return tensor

    x_vis = normalize(x_vis)
    event_feat_vis = normalize(event_feat_vis)
    output_x_vis = normalize(output_x_vis)
    if snr_map is not None:
        snr_map = normalize(snr_map)

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Visualize x (heatmap)
    im0 = axes[0].imshow(x_vis, cmap='viridis')
    axes[0].set_title('Input x (dim 0)')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    plt.colorbar(im0, ax=axes[0])

    # Visualize event_feat (heatmap)
    im1 = axes[1].imshow(event_feat_vis, cmap='viridis')
    axes[1].set_title('Event Feature (dim 0)')
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    plt.colorbar(im1, ax=axes[1])

    # Visualize snr_map (heatmap)
    if snr_map is not None:
        im2 = axes[2].imshow(snr_map, cmap='gray')
        axes[2].set_title('SNR Map (dim 0)')
        axes[2].set_xlabel('Width')
        axes[2].set_ylabel('Height')
        plt.colorbar(im2, ax=axes[2])
    else:
        axes[2].axis('off')

    # Visualize output_x (heatmap)
    im3 = axes[3].imshow(output_x_vis, cmap='viridis')
    axes[3].set_title('Output x (dim 0)')
    axes[3].set_xlabel('Width')
    axes[3].set_ylabel('Height')
    plt.colorbar(im3, ax=axes[3])

    # Manually adjust spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Save image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

def visualize_tensors(x, event_feat, atten, output_x, save_path="visualization.png", true_shape=None):
    """
    Visualize tensors x, event_feat, atten and output_x, display first dimension only, reshape to [H, W].

    Args:
        x: Initial input tensor, shape [batch, seq_len, embed_dim]
        event_feat: Event feature tensor, same shape as x
        atten: Attention map, same shape as x
        output_x: Final output tensor, same shape as x
        save_path: Path to save image, default "visualization.png"
        true_shape: True shape, shape [batch, 2], e.g., [[height, width]]
    """
    # Select first batch data
    batch_idx = 0
    if true_shape is None:
        raise ValueError("true_shape must be provided for reshaping")
    resize_shape = (true_shape[batch_idx][0].item() // 16, true_shape[batch_idx][1].item() // 16)
    H, W = resize_shape
    seq_len = H * W

    # Extract first batch data
    x = x[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    event_feat = event_feat[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    atten = atten[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    output_x = output_x[batch_idx].detach().cpu().numpy()  # [seq_len, embed_dim]
    print(f"x.shape: {x.shape}, event_feat.shape: {event_feat.shape}, atten.shape: {atten.shape}, output_x.shape: {output_x.shape}")

    # Verify seq_len matches
    if x.shape[0] != seq_len:
        raise ValueError(f"seq_len ({x.shape[0]}) does not match expected H*W ({seq_len})")

    # Reshape to [H, W], take first dimension
    x_vis = x[:, 100].reshape(H, W)  # [H, W]
    event_feat_vis = event_feat[:, 100].reshape(H, W)  # [H, W]
    atten_vis = atten[:, 100].reshape(H, W)  # [H, W]
    output_x_vis = output_x[:, 100].reshape(H, W)  # [H, W]

    # Normalize to [0, 1] for display
    def normalize(tensor):
        # tensor = tensor - tensor.min()
        # tensor = tensor / (tensor.max() + 1e-8)
        return tensor

    x_vis = normalize(x_vis)
    event_feat_vis = normalize(event_feat_vis)
    atten_vis = normalize(atten_vis)
    output_x_vis = normalize(output_x_vis)

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Visualize x (heatmap)
    im0 = axes[0].imshow(x_vis, cmap='viridis')
    axes[0].set_title('Input x (dim 0)')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    plt.colorbar(im0, ax=axes[0])

    # Visualize event_feat (heatmap)
    im1 = axes[1].imshow(event_feat_vis, cmap='viridis')
    axes[1].set_title('Event Feature (dim 0)')
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    plt.colorbar(im1, ax=axes[1])

    # Visualize atten (heatmap)
    im2 = axes[2].imshow(atten_vis, cmap='viridis')
    axes[2].set_title('Attention Map (dim 0)')
    axes[2].set_xlabel('Width')
    axes[2].set_ylabel('Height')
    plt.colorbar(im2, ax=axes[2])

    # Visualize output_x (heatmap)
    im3 = axes[3].imshow(output_x_vis, cmap='viridis')
    axes[3].set_title('Output x (dim 0)')
    axes[3].set_xlabel('Width')
    axes[3].set_ylabel('Height')
    plt.colorbar(im3, ax=axes[3])

    # Manually adjust spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Save image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

def visualize_image_snr(old_image, new_image, snr_map, save_path="image_snr_visualization.png"):
    """
    Visualize old_image, new_image and snr_map.

    Args:
        old_image: Original image tensor, shape [B, 3, H, W]
        new_image: Enhanced image tensor, shape [B, 3, H, W]
        snr_map: Signal-to-noise ratio map, shape [B, 1, H, W]
        save_path: Path to save image, default "image_snr_visualization.png"
    """
    # Select first batch data
    batch_idx = 0
    old_image = old_image[batch_idx].detach().cpu().numpy()  # [3, H, W]
    new_image = new_image[batch_idx].detach().cpu().numpy()  # [3, H, W]
    snr_map = snr_map[batch_idx, 0].detach().cpu().numpy()  # [H, W]

    # Convert image from [C, H, W] to [H, W, C] for display
    old_image = old_image.transpose(1, 2, 0)  # [H, W, 3]
    new_image = new_image.transpose(1, 2, 0)  # [H, W, 3]

    # Normalize to [0, 1] for display
    def normalize(tensor):
        tensor = tensor - tensor.min()
        tensor = tensor / (tensor.max() + 1e-8)
        return tensor

    old_image = normalize(old_image)
    new_image = normalize(new_image)
    snr_map = normalize(snr_map)

    # Create 1x3 subplot layout
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    # Visualize old_image (RGB image)
    axes[0].imshow(old_image)
    axes[0].set_title('Old Image')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    axes[0].axis('off')  # Hide axes

    # Visualize new_image (RGB image)
    axes[1].imshow(new_image)
    axes[1].set_title('New Image')
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    axes[1].axis('off')  # Hide axes

    # Visualize snr_map (grayscale heatmap)
    im2 = axes[2].imshow(snr_map, cmap='gray')
    axes[2].set_title('SNR Map')
    axes[2].set_xlabel('Width')
    axes[2].set_ylabel('Height')
    plt.colorbar(im2, ax=axes[2])

    # Manually adjust spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Save image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")
