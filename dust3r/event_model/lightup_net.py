import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint  # Import checkpoint functionality
# from einops import rearrange
import matplotlib.pyplot as plt
import os
import numpy as np

# Channel attention module, similar to eca_layer in EvLight
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# Spatial attention module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

# Basic convolution block with normalization and activation
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True, norm=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False) if activation else None
        self.norm = nn.InstanceNorm2d(out_channels) if norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

# Residual block with attention mechanism, similar to ECAResidualBlock in EvLight
class AttentionResidualBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1, norm=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1, norm=True)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Apply channel attention
        out = out * self.ca(out)
        # Apply spatial attention
        out = out * self.sa(out)
        
        # Avoid in-place operations
        out = out + residual
        return out

# SNR enhancement module, similar to SNR_enhance in EvLight
class SNREnhanceModule(nn.Module):
    def __init__(self, channels, snr_threshold=0.5, depth=1):
        super(SNREnhanceModule, self).__init__()
        self.channels = channels
        self.depth = depth
        self.threshold = snr_threshold
        
        # Reduce feature extractor depth and complexity
        self.img_extractors = nn.ModuleList([ConvLayer(channels, channels) for _ in range(depth)])
        self.ev_extractors = nn.ModuleList([ConvLayer(channels, channels) for _ in range(depth)])
        
        # Simplified feature fusion layer
        self.fusion = nn.Sequential(
            ConvLayer(channels*3, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, img_feat, event_feat, snr_map, att_feat):
        """
        Args:
            img_feat: Image features [B, C, H, W]
            event_feat: Event features [B, C, H, W]
            snr_map: SNR map [B, 1, H, W]
            att_feat: Attention features [B, C, H, W]
        """
        # Process SNR map based on threshold
        high_snr = snr_map.clone()
        high_snr[high_snr <= self.threshold] = 0.3
        high_snr[high_snr > self.threshold] = 0.7
        low_snr = 1 - high_snr
        
        # Expand SNR map to feature channel number
        high_snr_expanded = high_snr.repeat(1, self.channels, 1, 1)
        low_snr_expanded = low_snr.repeat(1, self.channels, 1, 1)
        
        # Simplified extraction process
        for i in range(self.depth):
            img_feat = self.img_extractors[i](img_feat)
            event_feat = self.ev_extractors[i](event_feat)
        
        # Select image features in high SNR regions
        high_snr_feat = torch.mul(img_feat, high_snr_expanded)
        
        # Select event features in low SNR regions
        low_snr_feat = torch.mul(event_feat, low_snr_expanded)
        
        # Feature fusion
        fused_feat = self.fusion(torch.cat([high_snr_feat, low_snr_feat, att_feat], dim=1))
        
        return fused_feat

# Adaptive feature fusion module
class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveFeatureFusion, self).__init__()
        
        # Spatial attention module for learning spatial weights
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Channel attention module for learning channel weights
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # SNR-guided feature fusion
        self.snr_fusion = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, img_feat, event_feat, snr_map=None):
        # Ensure feature map sizes are consistent
        if img_feat.shape[2:] != event_feat.shape[2:]:
            event_feat = F.interpolate(event_feat, size=img_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Compute spatial attention weights
        avg_pool = torch.mean(torch.cat([img_feat, event_feat], dim=1), dim=1, keepdim=True)
        max_pool, _ = torch.max(torch.cat([img_feat, event_feat], dim=1), dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        
        # Compute channel attention weights
        channel_weights = self.channel_attention(img_feat + event_feat)
        
        # If SNR map is available, perform SNR-guided fusion
        if snr_map is not None:
            # Ensure SNR map size is correct
            if snr_map.shape[2:] != img_feat.shape[2:]:
                snr_map = F.interpolate(snr_map, size=img_feat.shape[2:], mode='bilinear', align_corners=False)
            
            # Generate fusion weights using SNR map
            snr_weights = self.snr_fusion(snr_map)
            
            # Combine SNR weights for fusion
            fused_feat = img_feat * spatial_weights * channel_weights * snr_weights + \
                        event_feat * spatial_weights * channel_weights * (1 - snr_weights)
        else:
            # If no SNR map, use original fusion method
            fused_feat = img_feat * spatial_weights * channel_weights + \
                        event_feat * (1 - spatial_weights) * channel_weights
        
        # Final fusion
        out = self.fusion_conv(fused_feat)
        
        return out

# Improved image encoder
class ImageEncoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=16):
        """Lightweight image encoder
        
        Args:
            input_channels: Number of input channels
            base_channels: Base number of channels
        """
        super(ImageEncoder, self).__init__()
        self.conv1 = ConvLayer(input_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = ConvLayer(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1)
        
        self.conv3 = ConvLayer(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1)
        
        self.conv4 = ConvLayer(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        
        c2 = self.conv2(c1)
        
        c3 = self.conv3(c2)
        
        c4 = self.conv4(c3)
        
        return c1, c2, c3, c4

# Improved event encoder
class EventEncoder(nn.Module):
    def __init__(self, input_channels=5, base_channels=16):
        """Lightweight event encoder
        
        Args:
            input_channels: Number of input channels
            base_channels: Base number of channels
        """
        super(EventEncoder, self).__init__()
        self.conv1 = ConvLayer(input_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = ConvLayer(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1)
        
        self.conv3 = ConvLayer(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1)
        
        self.conv4 = ConvLayer(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1)
        
        # Upsampling path
        self.upconv3 = ConvLayer(base_channels*8, base_channels*4, kernel_size=3, stride=1, padding=1)
        
        self.upconv2 = ConvLayer(base_channels*4, base_channels*2, kernel_size=3, stride=1, padding=1)
        
        self.upconv1 = ConvLayer(base_channels*2, base_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Downsampling path
        c1 = self.conv1(x)
        
        c2 = self.conv2(c1)
        
        c3 = self.conv3(c2)
        
        c4 = self.conv4(c3)
        
        # Upsampling path - avoid in-place operations
        up3 = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False)
        up3 = self.upconv3(up3)
        u3 = up3 + c3  # Non in-place operation
        
        up2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        up2 = self.upconv2(up2)
        u2 = up2 + c2  # Non in-place operation
        
        up1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        up1 = self.upconv1(up1)
        u1 = up1 + c1  # Non in-place operation
        
        return u1, u2, u3, c4

# Decoder with SNR enhancement module
class Decoder(nn.Module):
    def __init__(self, base_channels=16, snr_thresholds=[0.5, 0.5, 0.5]):
        super(Decoder, self).__init__()
        
        # Reduce feature fusion module complexity
        self.fusion4 = AdaptiveFeatureFusion(base_channels*8, base_channels*8)
        self.fusion3 = AdaptiveFeatureFusion(base_channels*4, base_channels*4)
        self.fusion2 = AdaptiveFeatureFusion(base_channels*2, base_channels*2)
        self.fusion1 = AdaptiveFeatureFusion(base_channels, base_channels)
        
        # Reduce SNR enhancement module depth
        self.snr_enhance4 = SNREnhanceModule(base_channels*8, snr_thresholds[2], depth=1)
        self.snr_enhance3 = SNREnhanceModule(base_channels*4, snr_thresholds[1], depth=1)
        self.snr_enhance2 = SNREnhanceModule(base_channels*2, snr_thresholds[0], depth=1)
        
        # Upsampling modules
        self.upconv3 = ConvLayer(base_channels*8, base_channels*4, kernel_size=3, stride=1, padding=1)
        self.upconv2 = ConvLayer(base_channels*4, base_channels*2, kernel_size=3, stride=1, padding=1)
        self.upconv1 = ConvLayer(base_channels*2, base_channels, kernel_size=3, stride=1, padding=1)
        
        # Output layer
        self.output_conv = nn.Sequential(
            ConvLayer(base_channels, 3, kernel_size=3, stride=1, padding=1, activation=False)
        )

    def generate_snr_map(self, image, blur_image, factor=10.0):
        """Generate SNR map with simplified calculation"""
        # Convert to grayscale using fixed weights to avoid complex computation
        gray = image.mean(dim=1, keepdim=True)
        gray_blur = blur_image.mean(dim=1, keepdim=True)
        
        # Compute noise
        noise = torch.abs(gray - gray_blur)
        
        # Compute SNR, add small value to avoid division by zero
        snr = torch.div(gray_blur, noise + 0.0001)
        
        # Normalize to [0, 1]
        batch_size = snr.shape[0]
        snr_max, _ = torch.max(snr.view(batch_size, -1), dim=1, keepdim=True)
        snr_max = snr_max.view(batch_size, 1, 1, 1)
        snr = snr * factor / (snr_max + 0.0001)
        snr = torch.clamp(snr, min=0, max=1.0)
        
        return snr

    def forward(self, img_features, event_features, low_light_img):
        img_f1, img_f2, img_f3, img_f4 = img_features
        ev_f1, ev_f2, ev_f3, ev_f4 = event_features
        
        # Generate blurred image for SNR calculation
        with torch.no_grad():
            blur_kernel = torch.ones(1, 1, 5, 5).to(low_light_img.device) / 25
            low_light_blur = F.conv2d(
                F.pad(low_light_img, (2, 2, 2, 2), mode='reflect'),
                blur_kernel.repeat(3, 1, 1, 1),
                groups=3
            )
        
        # Generate SNR map
        snr_map = self.generate_snr_map(low_light_img, low_light_blur)
        
        # Feature fusion
        f4 = self.fusion4(img_f4, ev_f4, F.interpolate(snr_map, size=img_f4.shape[2:]))
        f4 = self.snr_enhance4(img_f4, ev_f4, F.interpolate(snr_map, size=f4.shape[2:]), f4)
        
        # Upsample and fuse - avoid in-place operations
        up3 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        up3 = self.upconv3(up3)
        f3 = self.fusion3(img_f3, ev_f3, F.interpolate(snr_map, size=img_f3.shape[2:]))
        u3 = up3 + f3  # Non in-place operation
        u3 = self.snr_enhance3(img_f3, ev_f3, F.interpolate(snr_map, size=u3.shape[2:]), u3)
        
        up2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        up2 = self.upconv2(up2)
        f2 = self.fusion2(img_f2, ev_f2, F.interpolate(snr_map, size=img_f2.shape[2:]))
        u2 = up2 + f2  # Non in-place operation
        u2 = self.snr_enhance2(img_f2, ev_f2, F.interpolate(snr_map, size=u2.shape[2:]), u2)
        
        up1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        up1 = self.upconv1(up1)
        f1 = self.fusion1(img_f1, ev_f1, F.interpolate(snr_map, size=img_f1.shape[2:]))
        u1 = up1 + f1  # Non in-place operation
        
        # Generate enhanced image residual
        enhanced_residual = self.output_conv(u1)
        
        return enhanced_residual, snr_map

class EasyIlluminationNet(nn.Module):
    def __init__(self, image_channels=3, event_channels=None, verbose_viz=False):  # event_channels no longer used, kept for interface compatibility
        super(EasyIlluminationNet, self).__init__()
        self.verbose_viz = verbose_viz
        
        # Illumination feature extractor - simplified version
        self.ill_extractor = nn.Sequential(
            nn.Conv2d(
                image_channels + 1,  # Image + initial illumination map
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        
        # Illumination mapping - reduced to single channel
        self.reduce = nn.Sequential(
            nn.Conv2d(16, 1, 1, 1, 0),
            nn.Sigmoid()  # Ensure illumination map range is [0,1], more consistent with paper description
        )
        
    def _generate_illumination_map(self, img):
        """Generate illumination prior map"""
        return torch.max(img, dim=1, keepdim=True)[0]
        
    def _generate_snr_map(self, enhanced_img):
        """Generate SNR map - using simpler and more effective method"""
        # Convert to grayscale
        r, g, b = enhanced_img[:, 0:1], enhanced_img[:, 1:2], enhanced_img[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b  # Standard RGB to grayscale conversion
        
        # Denoise grayscale image using mean filtering
        kernel_size = 5
        padding = kernel_size // 2
        denoised_gray = F.avg_pool2d(
            F.pad(gray, (padding, padding, padding, padding), mode='reflect'),
            kernel_size=kernel_size, 
            stride=1
        )
        
        # Compute noise
        noise = torch.abs(gray - denoised_gray)
        epsilon = 1e-6  # Avoid division by zero
        
        # Compute SNR
        snr_map = torch.div(denoised_gray, noise + epsilon)
        
        # Use adaptive normalization
        batch_size = snr_map.shape[0]
        snr_flat = snr_map.view(batch_size, -1)
        
        # Use percentile for normalization to avoid outlier effects
        q1 = torch.quantile(snr_flat, 0.25, dim=1).view(batch_size, 1, 1, 1)
        q3 = torch.quantile(snr_flat, 0.75, dim=1).view(batch_size, 1, 1, 1)
        iqr = q3 - q1
        
        # Use IQR for normalization, more robust to outliers
        snr_map = (snr_map - q1) / (iqr + epsilon)
        
        # Use sigmoid function for smooth mapping
        snr_map = torch.sigmoid(snr_map)
        
        # Ensure no NaN values
        snr_map = torch.nan_to_num(snr_map, nan=0.5, posinf=1.0, neginf=0.0)
        
        return snr_map
    
    def forward(self, low_light_image, event_voxel=None):  # event_voxel parameter kept but not used
        # 1. Generate illumination prior map
        illumination_prior = self._generate_illumination_map(low_light_image)
        
        if self.verbose_viz:
            # Visualize illumination prior map
            save_dir = "visualization/lightup_net"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save original low-light image
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(low_light_image[0].permute(1,2,0).cpu().detach().numpy())
            plt.title('Low Light Image')
            plt.subplot(122)
            plt.imshow(illumination_prior[0].permute(1,2,0).cpu().detach().numpy())
            plt.title('Illumination Prior')
            plt.savefig(f"{save_dir}/step1_illumination_prior.png")
            plt.close()

            # Save Low Light Image
            plt.figure()
            plt.imshow(low_light_image[0].permute(1,2,0).cpu().detach().numpy())
            plt.axis('off')  # Turn off axes
            plt.savefig(f"{save_dir}/low_light_image.png", bbox_inches='tight', pad_inches=0)
            plt.close()

            # Save Illumination Prior
            plt.figure()
            plt.imshow(illumination_prior[0].permute(1,2,0).cpu().detach().numpy())
            plt.axis('off')  # Turn off axes
            plt.savefig(f"{save_dir}/illumination_prior.png", bbox_inches='tight', pad_inches=0)
            plt.close()
        
        # 2. Extract illumination features and predict illumination map
        pred_illu_feature = self.ill_extractor(torch.cat((low_light_image, illumination_prior), dim=1))
        illumination_map = self.reduce(pred_illu_feature)
        
        if self.verbose_viz:
            # Visualize illumination features and illumination map
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(pred_illu_feature[0,0].cpu().detach().numpy())
            plt.title('Illumination Feature (Channel 0)')
            plt.subplot(132)
            plt.imshow(illumination_map[0,0].cpu().detach().numpy())
            plt.title('Illumination Map')
            plt.subplot(133)
            plt.imshow(illumination_map[0,0].cpu().detach().numpy(), cmap='jet')
            plt.title('Illumination Map (Jet)')
            plt.savefig(f"{save_dir}/step2_illumination_map.png")
            plt.close()

            # Save I_illum illumination map
            plt.figure()
            plt.imshow(illumination_map[0].permute(1,2,0).cpu().detach().numpy())
            plt.axis('off')  # Turn off axes
            plt.savefig(f"{save_dir}/I_illum_illumination_map.png", bbox_inches='tight', pad_inches=0)
            plt.close()
        
        # 3. Preliminary image enhancement
        enhanced_image = low_light_image * illumination_map
        
        if self.verbose_viz:
            # Visualize enhanced image
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(low_light_image[0].permute(1,2,0).cpu().detach().numpy())
            plt.title('Low Light Image')
            plt.subplot(132)
            plt.imshow(enhanced_image[0].permute(1,2,0).cpu().detach().numpy())
            plt.title('Enhanced Image')
            plt.subplot(133)
            plt.imshow((enhanced_image[0] - low_light_image[0]).permute(1,2,0).cpu().detach().numpy())
            plt.title('Difference')
            plt.savefig(f"{save_dir}/step3_enhanced_image.png")
            plt.close()

            # Save I_enhanced enhanced image
            plt.figure()
            plt.imshow(enhanced_image[0].permute(1,2,0).cpu().detach().numpy())
            plt.axis('off')  # Turn off axes
            plt.savefig(f"{save_dir}/I_enhanced_enhanced_image.png", bbox_inches='tight', pad_inches=0)
            plt.close()   
        
        # 4. Generate SNR map
        snr_map = self._generate_snr_map(enhanced_image)
        
        if self.verbose_viz:
            # Visualize SNR map
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(snr_map[0,0].cpu().detach().numpy())
            plt.title('SNR Map')
            plt.subplot(132)
            plt.imshow(snr_map[0,0].cpu().detach().numpy(), cmap='jet')
            plt.title('SNR Map (Jet)')
            plt.subplot(133)
            plt.hist(snr_map[0,0].cpu().detach().numpy().flatten(), bins=50)
            plt.title('SNR Distribution')
            plt.savefig(f"{save_dir}/step4_snr_map.png")
            plt.close()

            # Save all intermediate results to numpy files
            np.save(f"{save_dir}/low_light_image.npy", low_light_image.cpu().detach().numpy())
            np.save(f"{save_dir}/illumination_prior.npy", illumination_prior.cpu().detach().numpy())
            np.save(f"{save_dir}/pred_illu_feature.npy", pred_illu_feature.cpu().detach().numpy())
            np.save(f"{save_dir}/illumination_map.npy", illumination_map.cpu().detach().numpy())
            np.save(f"{save_dir}/enhanced_image.npy", enhanced_image.cpu().detach().numpy())
            np.save(f"{save_dir}/snr_map.npy", snr_map.cpu().detach().numpy())
        
        return enhanced_image, snr_map

# Event-guided low-light image enhancement network
class ComplexEvLightEnhancer(nn.Module):
    def __init__(self, image_channels=3, event_channels=5, base_channels=16, snr_thresholds=[0.5, 0.5, 0.5], use_checkpoint=True):
        """Complex event-guided low-light image enhancement network
        
        Args:
            image_channels: Number of image input channels, default is 3 (RGB)
            event_channels: Number of event input channels, default is 5
            base_channels: Base number of channels, default is 16 (reduce memory usage)
            snr_thresholds: List of SNR thresholds
            use_checkpoint: Whether to use gradient checkpointing to reduce memory usage
        """
        super(ComplexEvLightEnhancer, self).__init__()
        
        # Use lighter encoders and decoder
        self.image_encoder = ImageEncoder(image_channels, base_channels)
        self.event_encoder = EventEncoder(event_channels, base_channels)
        
        # Add adaptive feature fusion modules
        self.fusion1 = AdaptiveFeatureFusion(base_channels, base_channels)
        self.fusion2 = AdaptiveFeatureFusion(base_channels*2, base_channels*2)
        self.fusion3 = AdaptiveFeatureFusion(base_channels*4, base_channels*4)
        self.fusion4 = AdaptiveFeatureFusion(base_channels*8, base_channels*8)
        
        self.decoder = Decoder(base_channels, snr_thresholds)
        self.use_checkpoint = use_checkpoint
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, low_light_image, event_voxel):
        # Normalize event data
        if event_voxel.max() > 1.0:
            event_voxel = event_voxel / 255.0 if event_voxel.max() > 255.0 else event_voxel / event_voxel.max()
        
        # Use checkpoint to reduce memory usage
        if self.use_checkpoint and self.training:
            # Encode image and event data
            img_features = checkpoint.checkpoint(self.image_encoder, low_light_image)
            event_features = checkpoint.checkpoint(self.event_encoder, event_voxel)
            
            # Adaptive feature fusion
            fused_features = [
                self.fusion1(img_features[0], event_features[0]),
                self.fusion2(img_features[1], event_features[1]),
                self.fusion3(img_features[2], event_features[2]),
                self.fusion4(img_features[3], event_features[3])
            ]
            
            # Decode fused features to generate enhanced image residual
            enhanced_residual, snr_map = checkpoint.checkpoint(self.decoder, fused_features, fused_features, low_light_image)
        else:
            # Without checkpoint
            img_features = self.image_encoder(low_light_image)
            event_features = self.event_encoder(event_voxel)
            
            # Adaptive feature fusion
            fused_features = [
                self.fusion1(img_features[0], event_features[0]),
                self.fusion2(img_features[1], event_features[1]),
                self.fusion3(img_features[2], event_features[2]),
                self.fusion4(img_features[3], event_features[3])
            ]
            
            enhanced_residual, snr_map = self.decoder(fused_features, fused_features, low_light_image)
        
        # Residual connection, enhanced image is original image plus residual
        enhanced_image = torch.clamp(low_light_image + enhanced_residual, 0, 1)
        
        return enhanced_image, snr_map

# Unified interface, select different enhancement methods based on mode
class EvLightEnhancer(nn.Module):
    def __init__(self, mode='complex', image_channels=3, event_channels=5, base_channels=16, snr_thresholds=[0.5, 0.5, 0.5], use_checkpoint=True):
        """
        Unified interface for low-light image enhancement network

        Args:
            mode: Enhancement mode, 'none' for no enhancement, 'easy' for improved lightweight enhancement (without event data), 'complex' for complex enhancement network
            image_channels: Number of image input channels
            event_channels: Number of event input channels
            base_channels: Base number of channels
            snr_thresholds: List of SNR thresholds
            use_checkpoint: Whether to use gradient checkpointing
        """
        super(EvLightEnhancer, self).__init__()
        self.mode = mode

        if mode == 'easy':
            # Use improved easy enhancement network (does not depend on event data)
            self.illumination_net = EasyIlluminationNet(image_channels)
        elif mode == 'complex':
            self.enhancer = ComplexEvLightEnhancer(image_channels, event_channels, base_channels, snr_thresholds, use_checkpoint)

    def forward(self, low_light_image, event_voxel=None):
        """
        Forward pass

        Args:
            low_light_image: Low-light image [B, C, H, W]
            event_voxel: Event voxel data [B, C, H, W], ignored for 'none' and 'easy' modes

        Returns:
            enhanced_image: Enhanced image [B, C, H, W]
            snr_map: Signal-to-noise ratio map [B, 1, H, W]
        """
        if self.mode == 'none':
            batch_size, _, height, width = low_light_image.shape
            return low_light_image, torch.zeros((batch_size, 1, height, width), device=low_light_image.device)

        if self.mode == 'easy':
            # Use improved easy enhancement network, without event data
            enhanced_image, snr_map = self.illumination_net(low_light_image)
            return enhanced_image, snr_map

        # Complex mode
        return self.enhancer(low_light_image, event_voxel)

# Test code
if __name__ == '__main__':
    batch_size, height, width = 2, 256, 256
    low_light_image = torch.rand(batch_size, 3, height, width)
    event_voxel = torch.rand(batch_size, 5, height, width)

    modes = ['none', 'easy', 'complex']

    for mode in modes:
        print(f"\nTesting {mode} mode:")
        model = EvLightEnhancer(mode=mode)
        if mode == 'complex':
            print("Using event data for enhancement...")
            enhanced_image, snr_map = model(low_light_image, event_voxel)
        else:
            if mode == 'easy':
                print("Not using event data, only based on improved easy enhancement...")
            else:
                print("No enhancement performed...")
            enhanced_image, snr_map = model(low_light_image, None)

        print(f"Input image shape: {low_light_image.shape}")
        if mode == 'complex':
            print(f"Event data shape: {event_voxel.shape}")
        print(f"Enhanced image shape: {enhanced_image.shape}")
        print(f"SNR map shape: {snr_map.shape}")