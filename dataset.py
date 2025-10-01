"""
dataset.py - Dataset classes for video-based scene completion

Supports:
1. Video-only input (no poses needed!)
2. Optional depth data
3. Data augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VideoSceneDataset(Dataset):
    """
    Dataset for video-based scene completion
    No pose data required!
    """
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 num_frames: int = 10,
                 img_size: int = 224,
                 voxel_resolution: int = 64,
                 use_augmentation: bool = True):
        """
        Args:
            data_dir: Root directory with videos
            split: 'train', 'val', or 'test'
            num_frames: Number of frames to sample per video
            img_size: Image size for model input
            voxel_resolution: Resolution of output voxel grid
            use_augmentation: Apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.voxel_resolution = voxel_resolution
        
        # Load video list
        self.videos = self.load_video_list()
        
        # Setup augmentation
        if use_augmentation and split == 'train':
            self.transform = A.Compose([
                A.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def load_video_list(self):
        """Load list of video files"""
        split_file = self.data_dir / f'{self.split}_list.txt'
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                videos = [self.data_dir / line.strip() for line in f]
        else:
            # Automatically find all videos
            videos = list(self.data_dir.glob('**/*.mp4'))
            videos.extend(list(self.data_dir.glob('**/*.avi')))
            
            # Split into train/val
            if self.split == 'train':
                videos = videos[:int(len(videos) * 0.9)]
            else:
                videos = videos[int(len(videos) * 0.9):]
        
        return videos
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        """
        Load video and create training sample
        
        Returns:
            dict with keys:
                'video': (T, 3, H, W) video frames
                'video_path': path to video file
                'gt_voxels': (R, R, R) ground truth voxels (if available)
        """
        video_path = self.videos[idx]
        
        # Load video frames
        frames = self.load_video(video_path)
        
        # Apply augmentation
        frames = [self.transform(image=frame)['image'] for frame in frames]
        frames = torch.stack(frames)  # (T, 3, H, W)
        
        sample = {
            'video': frames,
            'video_path': str(video_path)
        }
        
        # Load ground truth voxels if available
        gt_path = video_path.parent / f"{video_path.stem}_voxels.npz"
        if gt_path.exists():
            data = np.load(gt_path)
            sample['gt_voxels'] = torch.from_numpy(data['voxels']).float()
            if 'sdf' in data:
                sample['gt_sdf'] = torch.from_numpy(data['sdf']).float()
        
        return sample
    
    def load_video(self, video_path: Path) -> List[np.ndarray]:
        """
        Load video frames
        
        Args:
            video_path: Path to video file
        Returns:
            frames: List of RGB frames (numpy arrays)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Sample frame indices uniformly
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            # Pad if needed
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Use last valid frame if read fails
                frames.append(frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8))
        
        cap.release()
        
        return frames


class RealSenseDataset(VideoSceneDataset):
    """
    Dataset specifically for RealSense D455 data
    Handles both RGB and depth streams
    """
    def __init__(self, *args, use_depth=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_depth = use_depth
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        if self.use_depth:
            # Load corresponding depth frames
            video_path = self.videos[idx]
            depth_frames = self.load_depth(video_path)
            
            if depth_frames is not None:
                sample['depth'] = depth_frames
        
        return sample
    
    def load_depth(self, video_path: Path) -> Optional[torch.Tensor]:
        """Load depth frames corresponding to RGB video"""
        # Assume depth is in a parallel directory structure
        depth_path = Path(str(video_path).replace('colorFrames', 'depthFrames'))
        
        if not depth_path.exists():
            # Try finding depth frames in same directory
            depth_dir = video_path.parent / 'depth'
            if depth_dir.exists():
                depth_frames = sorted(depth_dir.glob('*.png'))[:self.num_frames]
                
                depths = []
                for df in depth_frames:
                    depth = cv2.imread(str(df), cv2.IMREAD_ANYDEPTH)
                    depth = depth.astype(np.float32) / 1000.0  # mm to meters
                    depth = cv2.resize(depth, (self.img_size, self.img_size))
                    depths.append(depth)
                
                return torch.from_numpy(np.stack(depths)).unsqueeze(1)  # (T, 1, H, W)
        
        return None


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for quick testing
    Generates random video + voxel pairs
    """
    def __init__(self, 
                 num_samples=1000,
                 num_frames=10,
                 img_size=224,
                 voxel_resolution=64):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.img_size = img_size
        self.voxel_resolution = voxel_resolution
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random video frames
        video = torch.randn(self.num_frames, 3, self.img_size, self.img_size)
        
        # Random voxel grid (with some structure)
        center = self.voxel_resolution // 2
        R = self.voxel_resolution
        
        # Create a simple sphere
        x = torch.linspace(-1, 1, R)
        y = torch.linspace(-1, 1, R)
        z = torch.linspace(-1, 1, R)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        radius = 0.3 + 0.2 * torch.rand(1).item()
        voxels = ((X**2 + Y**2 + Z**2) < radius**2).float()
        
        # Add some noise
        voxels += 0.1 * torch.randn_like(voxels)
        voxels = torch.clamp(voxels, 0, 1)
        
        # SDF (signed distance)
        sdf = torch.sqrt(X**2 + Y**2 + Z**2) - radius
        
        return {
            'video': video,
            'gt_voxels': voxels,
            'gt_sdf': sdf
        }


def create_dataloaders(config):
    """
    Create train and validation dataloaders
    
    Args:
        config: Configuration dictionary
    Returns:
        train_loader, val_loader
    """
    data_type = config.get('data_type', 'video')
    
    if data_type == 'synthetic':
        # Use synthetic data for quick testing
        train_dataset = SyntheticDataset(
            num_samples=1000,
            num_frames=config['num_frames'],
            img_size=config['img_size'],
            voxel_resolution=config['voxel_resolution']
        )
        val_dataset = SyntheticDataset(
            num_samples=100,
            num_frames=config['num_frames'],
            img_size=config['img_size'],
            voxel_resolution=config['voxel_resolution']
        )
    
    elif data_type == 'realsense':
        # RealSense data with depth
        train_dataset = RealSenseDataset(
            data_dir=config['data_dir'],
            split='train',
            num_frames=config['num_frames'],
            img_size=config['img_size'],
            voxel_resolution=config['voxel_resolution'],
            use_depth=config.get('use_depth', True)
        )
        val_dataset = RealSenseDataset(
            data_dir=config['data_dir'],
            split='val',
            num_frames=config['num_frames'],
            img_size=config['img_size'],
            voxel_resolution=config['voxel_resolution'],
            use_depth=config.get('use_depth', True)
        )
    
    else:
        # Regular video data
        train_dataset = VideoSceneDataset(
            data_dir=config['data_dir'],
            split='train',
            num_frames=config['num_frames'],
            img_size=config['img_size'],
            voxel_resolution=config['voxel_resolution']
        )
        val_dataset = VideoSceneDataset(
            data_dir=config['data_dir'],
            split='val',
            num_frames=config['num_frames'],
            img_size=config['img_size'],
            voxel_resolution=config['voxel_resolution']
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader