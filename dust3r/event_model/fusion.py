import torch
import torch.nn as nn
from torch.distributed import all_reduce, ReduceOp
import math
from ..visualization import visualize_tensors, visualize_tensors_snr
import torch.nn.functional as F
import matplotlib.pyplot as plt

def check_shape_consistency(tensor, name):
    # Get tensor shape and convert to tensor
    shape = torch.tensor(tensor.shape, device=tensor.device, dtype=torch.long)  # e.g., [2, 768, 9, 16]
    
    # Create copies for aggregation
    shape_max = shape.clone()
    shape_min = shape.clone()
    
    # Use all_reduce to aggregate shapes across all ranks (take max and min)
    all_reduce(shape_max, op=ReduceOp.MAX)
    all_reduce(shape_min, op=ReduceOp.MIN)
    
    # Check if max and min are the same (i.e., if shapes are consistent)
    shape_list = shape.tolist()
    if torch.equal(shape_max, shape_min):
        print(f"{name} shape consistency: {shape_list} across all ranks")
    else:
        raise RuntimeError(f"{name} shape inconsistency: {shape_list} on rank {torch.distributed.get_rank()}, "
                         f"max shape: {shape_max.tolist()}, min shape: {shape_min.tolist()}")

class CrossAttention(nn.Module):
    """Multi-head cross-attention module for event features attending to image features"""
    
    def __init__(self, dim_q, dim_kv, dim_embed, dim_out, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., rope=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_embed // num_heads
        self.scale = head_dim ** -0.5
        self.dim_embed = dim_embed

        self.q_proj = nn.Linear(dim_q, dim_embed, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_kv, dim_embed, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_kv, dim_embed, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_embed, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        
    def forward(self, query, key, value, query_pos=None, key_pos=None, return_attention_weights=False):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        
        # Project query, key, and value
        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.dim_embed // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, self.dim_embed // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, self.dim_embed // self.num_heads).permute(0, 2, 1, 3)
        
        # Apply positional encoding (if available)
        if self.rope is not None and query_pos is not None and key_pos is not None:
            # Check rope object interface
            if hasattr(self.rope, 'apply_rotary'):
                q = self.rope.apply_rotary(q, positions=query_pos)
                k = self.rope.apply_rotary(k, positions=key_pos)
            
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention weights
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, self.dim_embed)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention_weights:
            # Return attention weights (average across all heads)
            attn_weights = attn.mean(dim=1)  # [B, Nq, Nk]
            return x, attn_weights
        return x

class ImageEventFusion(nn.Module):
    def __init__(self, event_channels=768, target_channels=1024):
        super().__init__()
        self.conv_adjust = nn.Conv2d(event_channels, target_channels, kernel_size=1)  # Adjust channel number
        self.attention = nn.MultiheadAttention(embed_dim=target_channels, num_heads=8)
        self.norm = nn.LayerNorm(target_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, event_feat, true_shape, snr_map=None,event_blk_idx=None):
        h, w = true_shape[0][0].item(), true_shape[0][1].item()
        scale = int(math.sqrt(h * w / x.shape[1]))
        target_size = (h // scale, w // scale)
        event_feat = F.interpolate(event_feat, size=target_size, mode='bilinear', align_corners=False)

        event_feat = self.conv_adjust(event_feat)

        B, C, H, W = event_feat.size()
        event_feat = event_feat.view(B, C, H * W).transpose(1, 2)

        old_x = x.clone()
        old_event_feat = event_feat.clone()

        if snr_map is not None:
            snr_map = F.interpolate(snr_map, size=target_size, mode='bilinear', align_corners=False)
            snr_map = snr_map.view(B, 1, H * W).transpose(1, 2)  # [B, H*W, 1]
            snr_weight = self.sigmoid(snr_map)

            x = x * snr_weight
            event_feat = event_feat*(1 - snr_weight)

        # Cross-Attention
        # x as query, event_feat as key and value
        # For MultiheadAttention, adjust dimensions to [seq_len, batch, embed_dim]
        x = x.transpose(0, 1)  # [576, 2, 1024]
        event_feat = event_feat.transpose(0, 1)  # [576, 2, 1024]

        # Compute attention
        attn_output, _ = self.attention(query=x, key=event_feat, value=event_feat)

        # Restore dimensions
        attn_output = attn_output.transpose(0, 1)  # [2, 576, 1024]

        # Residual connection and normalization
        x = self.norm(x.transpose(0, 1) + attn_output)  # [2, 576, 1024]

        if self.training and hasattr(self, 'verbose_viz') and self.verbose_viz:
            visualize_tensors_snr(old_x, old_event_feat, snr_map, attn_output, 
                                 save_path=f"visualization/tensorSNR{event_blk_idx}_visualization.png",
                                 true_shape=true_shape)
        return x

class EventImageFusion(nn.Module):
    def __init__(self, event_channels=768, target_channels=1024):
        super().__init__()
        # Multi-head attention
        self.attention = CrossAttention(
            dim_q=event_channels, 
            dim_kv=target_channels, 
            dim_embed=256, 
            dim_out=event_channels,
            num_heads=8
        )
        
        # Feature normalization layers
        self.norm = nn.LayerNorm(event_channels)
        self.image_norm = nn.LayerNorm(target_channels)
        
        # Feature enhancement module
        self.feature_enhance = nn.Sequential(
            nn.Linear(event_channels, event_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(event_channels * 2, event_channels),
            nn.LayerNorm(event_channels)
        )
        
        # Attention weights for visualization
        self.attention_weights = None
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        def _init_module(module):
            if isinstance(module, nn.Linear):
                # Use xavier initialization, more suitable for attention mechanisms
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize all modules
        self.apply(_init_module)
        
        # Specifically initialize attention module
        nn.init.xavier_uniform_(self.attention.q_proj.weight)
        nn.init.xavier_uniform_(self.attention.k_proj.weight)
        nn.init.xavier_uniform_(self.attention.v_proj.weight)
        nn.init.xavier_uniform_(self.attention.proj.weight)
        
        # Initialize attention module biases
        if self.attention.q_proj.bias is not None:
            nn.init.zeros_(self.attention.q_proj.bias)
        if self.attention.k_proj.bias is not None:
            nn.init.zeros_(self.attention.k_proj.bias)
        if self.attention.v_proj.bias is not None:
            nn.init.zeros_(self.attention.v_proj.bias)
        if self.attention.proj.bias is not None:
            nn.init.zeros_(self.attention.proj.bias)

    def forward(self, x, event_feat):
        # Feature normalization
        x = self.image_norm(x)
        event_feat = self.norm(event_feat)
        
        # Save original features for residual connection
        residual = event_feat
        
        # Compute attention
        attn_output, self.attention_weights = self.attention(
            query=event_feat, 
            key=x, 
            value=x,
            return_attention_weights=True
        )
        
        # Feature enhancement
        enhanced_feat = self.feature_enhance(attn_output)
        
        # Residual connection
        output = residual + enhanced_feat
        
        # Final normalization
        output = self.norm(output)
        
        return output
    
    def visualize_attention(self, save_path=None):
        """Visualize attention weights"""
        if self.attention_weights is not None and save_path is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.attention_weights.detach().cpu().numpy())
            plt.colorbar()
            plt.savefig(save_path)
            plt.close()
    
    def get_feature_stats(self, x, event_feat):
        """Get feature statistics"""
        stats = {
            'x_mean': x.mean().item(),
            'x_std': x.std().item(),
            'event_feat_mean': event_feat.mean().item(),
            'event_feat_std': event_feat.std().item(),
            'attention_weights_mean': self.attention_weights.mean().item() if self.attention_weights is not None else 0
        }
        return stats

class FeatureFusionLayer(nn.Module):
    """Feature fusion layer supporting multiple fusion strategies"""
    
    def __init__(self, dim, fusion_type='attention', num_heads=8):
        super().__init__()
        self.dim = dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # Attention fusion
            self.attention = CrossAttention(
                dim_q=dim,
                dim_kv=dim,
                dim_embed=dim,
                dim_out=dim,
                num_heads=num_heads
            )
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            
        elif fusion_type == 'concat':
            # Concatenation fusion
            self.fusion_conv = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
            )
            
        elif fusion_type == 'adaptive':
            # Adaptive fusion
            self.fusion_weights = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, 2),
                nn.Softmax(dim=-1)
            )
            
        # Feature enhancement
        if fusion_type == 'concat':
            dim_out = dim * 2
        else:
            dim_out = dim
        self.enhance = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.LayerNorm(dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim),
            nn.LayerNorm(dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.apply(_init_module)
        
    def forward(self, x1, x2, snr_map=None):
        """
        Fuse two features
        
        Args:
            x1: First feature [B, L, C]
            x2: Second feature [B, L, C]
            snr_map: SNR map [B, L, 1] or None
            
        Returns:
            Fused feature [B, L, C]
        """
        if self.fusion_type == 'attention':
            # Attention fusion
            x1 = self.norm1(x1)
            x2 = self.norm2(x2)
            
            # Compute attention
            attn_output = self.attention(query=x1, key=x2, value=x2)
            
            # Residual connection
            fused = x1 + attn_output
            
        elif self.fusion_type == 'concat':
            # Concatenation fusion
            concat_feat = torch.cat([x1, x2], dim=-1)
            fused = self.fusion_conv(concat_feat)
            
        elif self.fusion_type == 'adaptive':
            # Adaptive fusion
            concat_feat = torch.cat([x1, x2], dim=-1)
            weights = self.fusion_weights(concat_feat)  # [B, L, 2]
            fused = weights[:, :, 0:1] * x1 + weights[:, :, 1:2] * x2
        elif self.fusion_type == 'add':
            fused = x1 + x2
            
        # If SNR map is provided, use it to adjust fusion weights
        if snr_map is not None:
            snr_weight = torch.sigmoid(snr_map)
            fused = fused * snr_weight + x1 * (1 - snr_weight)
            
        # Feature enhancement
        enhanced = self.enhance(fused)
        
        # Residual connection
        output = fused + enhanced
        
        return output