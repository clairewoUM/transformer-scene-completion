"""
model.py - Pose-Free Transformer for 3D Scene Completion

This model learns camera poses implicitly from video frames without requiring
external pose estimation (SLAM, IMU, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

# ============================================================================
# POSE ESTIMATION MODULE (Learned from Video)
# ============================================================================

class PoseEstimationNetwork(nn.Module):
    """
    Estimate relative camera poses from frame pairs
    No external pose data needed!
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # Feature extraction for pose estimation
        self.conv1 = nn.Conv2d(6, 64, 7, stride=2, padding=3)  # Concatenated frame pair
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Predict rotation (as 6D representation for continuity)
        self.rotation_head = nn.Linear(512, 6)
        
        # Predict translation
        self.translation_head = nn.Linear(512, 3)
        
    def forward(self, frame1, frame2):
        """
        Estimate relative pose from frame1 to frame2
        
        Args:
            frame1, frame2: (B, 3, H, W) RGB frames
        Returns:
            rotation: (B, 3, 3) rotation matrix
            translation: (B, 3) translation vector
        """
        # Concatenate frames
        x = torch.cat([frame1, frame2], dim=1)  # (B, 6, H, W)
        
        # Extract features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, 512)
        
        # Predict 6D rotation representation
        rot_6d = self.rotation_head(x)  # (B, 6)
        rotation = self.rotation_6d_to_matrix(rot_6d)
        
        # Predict translation
        translation = self.translation_head(x)  # (B, 3)
        
        return rotation, translation
    
    @staticmethod
    def rotation_6d_to_matrix(rot_6d):
        """
        Convert 6D rotation representation to 3x3 rotation matrix
        Based on "On the Continuity of Rotation Representations in Neural Networks"
        """
        batch_size = rot_6d.shape[0]
        
        # First two columns
        a1 = rot_6d[:, :3]
        a2 = rot_6d[:, 3:6]
        
        # Normalize
        b1 = F.normalize(a1, dim=1)
        b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1)
        
        # Third column (cross product)
        b3 = torch.cross(b1, b2, dim=1)
        
        # Stack into rotation matrix
        rotation = torch.stack([b1, b2, b3], dim=2)  # (B, 3, 3)
        
        return rotation


# ============================================================================
# VIDEO TRANSFORMER ENCODER
# ============================================================================

class VideoTransformerEncoder(nn.Module):
    """
    Encode video frames with self-attention
    Captures temporal and spatial relationships
    """
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding (spatial)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Temporal embedding
        self.temporal_embed = nn.Parameter(torch.zeros(1, 100, embed_dim))  # Max 100 frames
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, video):
        """
        Args:
            video: (B, T, C, H, W) video frames
        Returns:
            features: (B, T*N, D) encoded features
            (T=num_frames, N=num_patches, D=embed_dim)
        """
        B, T, C, H, W = video.shape
        
        # Reshape to process all frames
        video = video.view(B * T, C, H, W)
        
        # Patch embedding
        x = self.patch_embed(video)  # (B*T, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, D)
        
        # Add spatial positional embedding
        x = x + self.pos_embed
        
        # Reshape back to (B, T, N, D)
        x = x.view(B, T, self.num_patches, -1)
        
        # Add temporal embedding
        x = x + self.temporal_embed[:, :T, :].unsqueeze(2)
        
        # Flatten temporal and spatial dimensions
        x = x.view(B, T * self.num_patches, -1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with multi-head attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 3D SCENE DECODER
# ============================================================================

class Scene3DDecoder(nn.Module):
    """
    Decode video features to 3D voxel grid
    Uses cross-attention between 3D queries and video features
    """
    def __init__(self,
                 voxel_resolution=64,
                 embed_dim=768,
                 depth=6,
                 num_heads=12):
        super().__init__()
        
        self.voxel_resolution = voxel_resolution
        
        # 3D voxel queries (learnable)
        self.voxel_queries = nn.Parameter(
            torch.randn(1, voxel_resolution**3, embed_dim)
        )
        
        # 3D positional encoding
        self.pos_encoding_3d = self.build_3d_pos_encoding(voxel_resolution, embed_dim)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(depth)
        ])
        
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(depth)
        ])
        
        self.norms1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.norms3 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            for _ in range(depth)
        ])
        
        # Output heads
        self.occupancy_head = nn.Linear(embed_dim, 1)
        self.sdf_head = nn.Linear(embed_dim, 1)
        
    def build_3d_pos_encoding(self, resolution, dim):
        """Build 3D positional encoding"""
        pos_encoding = torch.zeros(resolution**3, dim)
        
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    idx = i * resolution**2 + j * resolution + k
                    
                    # Use different frequency bands for x, y, z
                    for d in range(dim // 6):
                        freq = 2 ** d
                        pos_encoding[idx, d*6:d*6+2] = torch.tensor([
                            math.sin(freq * i / resolution),
                            math.cos(freq * i / resolution)
                        ])
                        pos_encoding[idx, d*6+2:d*6+4] = torch.tensor([
                            math.sin(freq * j / resolution),
                            math.cos(freq * j / resolution)
                        ])
                        pos_encoding[idx, d*6+4:d*6+6] = torch.tensor([
                            math.sin(freq * k / resolution),
                            math.cos(freq * k / resolution)
                        ])
        
        return nn.Parameter(pos_encoding.unsqueeze(0), requires_grad=False)
    
    def forward(self, video_features):
        """
        Args:
            video_features: (B, T*N, D) encoded video features
        Returns:
            occupancy: (B, R, R, R) occupancy grid
            sdf: (B, R, R, R) signed distance field
        """
        B = video_features.shape[0]
        
        # Initialize 3D queries
        queries = self.voxel_queries.expand(B, -1, -1)
        queries = queries + self.pos_encoding_3d
        
        # Apply cross-attention and self-attention layers
        for i in range(len(self.cross_attn_layers)):
            # Cross-attention to video features
            queries = queries + self.cross_attn_layers[i](
                self.norms1[i](queries),
                self.norms1[i](video_features),
                self.norms1[i](video_features)
            )[0]
            
            # Self-attention between voxels
            queries = queries + self.self_attn_layers[i](
                self.norms2[i](queries),
                self.norms2[i](queries),
                self.norms2[i](queries)
            )[0]
            
            # MLP
            queries = queries + self.mlps[i](self.norms3[i](queries))
        
        # Predict occupancy and SDF
        occupancy = torch.sigmoid(self.occupancy_head(queries))
        sdf = torch.tanh(self.sdf_head(queries))
        
        # Reshape to 3D grid
        R = self.voxel_resolution
        occupancy = occupancy.view(B, R, R, R)
        sdf = sdf.view(B, R, R, R)
        
        return occupancy, sdf


# ============================================================================
# COMPLETE MODEL
# ============================================================================

class PoseFreeSceneCompletion(nn.Module):
    """
    Complete pose-free scene completion model
    
    Pipeline:
    1. Encode video frames → features
    2. Estimate relative poses (learned)
    3. Decode to 3D scene
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 voxel_resolution=64,
                 embed_dim=512,
                 encoder_depth=8,
                 decoder_depth=4,
                 num_heads=8):
        super().__init__()
        
        # Video encoder
        self.video_encoder = VideoTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads
        )
        
        # Pose estimator (optional, learned from data)
        self.pose_estimator = PoseEstimationNetwork(feature_dim=embed_dim)
        
        # 3D scene decoder
        self.scene_decoder = Scene3DDecoder(
            voxel_resolution=voxel_resolution,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads
        )
        
    def forward(self, video, return_poses=False):
        """
        Args:
            video: (B, T, 3, H, W) video frames
            return_poses: whether to return estimated poses
        Returns:
            occupancy: (B, R, R, R) occupancy grid
            sdf: (B, R, R, R) signed distance field
            poses: (B, T, 4, 4) estimated camera poses (if return_poses=True)
        """
        B, T, C, H, W = video.shape
        
        # Encode video
        video_features = self.video_encoder(video)  # (B, T*N, D)
        
        # Optionally estimate poses
        poses = None
        if return_poses:
            poses = self.estimate_poses(video)
        
        # Decode to 3D scene
        occupancy, sdf = self.scene_decoder(video_features)
        
        if return_poses:
            return occupancy, sdf, poses
        else:
            return occupancy, sdf
    
    def estimate_poses(self, video):
        """
        Estimate camera poses for all frames
        
        Args:
            video: (B, T, 3, H, W)
        Returns:
            poses: (B, T, 4, 4) pose matrices
        """
        B, T, C, H, W = video.shape
        
        # Start with identity for first frame
        poses = [torch.eye(4, device=video.device).unsqueeze(0).expand(B, -1, -1)]
        
        # Estimate relative poses between consecutive frames
        for t in range(T - 1):
            R, t_vec = self.pose_estimator(video[:, t], video[:, t + 1])
            
            # Construct 4x4 transformation matrix
            pose = torch.eye(4, device=video.device).unsqueeze(0).expand(B, -1, -1).clone()
            pose[:, :3, :3] = R
            pose[:, :3, 3] = t_vec
            
            # Accumulate pose
            poses.append(poses[-1] @ pose)
        
        poses = torch.stack(poses, dim=1)  # (B, T, 4, 4)
        
        return poses


# ============================================================================
# LIGHTWEIGHT VERSION (For faster training)
# ============================================================================

class LightweightSceneCompletion(nn.Module):
    """
    Lightweight version for quick experiments
    ~10M parameters instead of ~100M
    """
    def __init__(self, voxel_resolution=64):
        super().__init__()
        
        self.voxel_resolution = voxel_resolution
        
        # Simple CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Temporal aggregation
        self.temporal_conv = nn.Conv1d(256 * 64, 512, kernel_size=3, padding=1)
        
        # 3D decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, voxel_resolution ** 3 * 2)  # *2 for occupancy + sdf
        )
    
    def forward(self, video):
        B, T, C, H, W = video.shape
        
        # Encode frames
        video_flat = video.view(B * T, C, H, W)
        features = self.encoder(video_flat)  # (B*T, 256, 8, 8)
        features = features.view(B, T, -1)  # (B, T, 256*64)
        
        # Temporal aggregation
        features = features.transpose(1, 2)  # (B, 256*64, T)
        features = self.temporal_conv(features)  # (B, 512, T)
        features = features.mean(dim=2)  # (B, 512)
        
        # Decode to 3D
        output = self.decoder(features)  # (B, R^3 * 2)
        
        R = self.voxel_resolution
        occupancy = torch.sigmoid(output[:, :R**3].view(B, R, R, R))
        sdf = torch.tanh(output[:, R**3:].view(B, R, R, R))
        
        return occupancy, sdf