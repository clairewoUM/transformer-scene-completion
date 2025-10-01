"""
utils.py - Utility functions for scene completion
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import trimesh
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class VideoProcessor:
    """
    Process video files for model input
    """
    def __init__(self, img_size=224, num_frames=10):
        self.img_size = img_size
        self.num_frames = num_frames
        
        # Normalization (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def load_video(self, video_path: str) -> torch.Tensor:
        """
        Load video and prepare for model
        
        Args:
            video_path: Path to video file
        Returns:
            video: (T, 3, H, W) tensor
        """
        cap = cv2.VideoCapture(str(video_path))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Sample frames uniformly
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
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
        
        cap.release()
        
        # Prepare frames
        video = self.prepare_frames(frames)
        
        return video
    
    def prepare_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Prepare list of frames for model
        
        Args:
            frames: List of RGB frames (numpy arrays)
        Returns:
            video: (T, 3, H, W) tensor
        """
        processed = []
        
        for frame in frames:
            # Resize
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            
            # To tensor
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            # Normalize
            frame = (frame - self.mean) / self.std
            
            processed.append(frame)
        
        video = torch.stack(processed)  # (T, 3, H, W)
        
        return video


class MeshExporter:
    """
    Export voxel grids to meshes
    """
    def __init__(self, voxel_size=0.05):
        self.voxel_size = voxel_size
    
    def extract_mesh(self, 
                     voxels: np.ndarray, 
                     threshold: float = 0.5,
                     use_sdf: bool = False) -> Dict:
        """
        Extract mesh from voxel grid using marching cubes
        
        Args:
            voxels: (R, R, R) voxel grid
            threshold: Threshold for marching cubes
            use_sdf: Whether voxels represent SDF
        Returns:
            mesh: Dictionary with 'vertices', 'faces', 'normals'
        """
        # Run marching cubes
        if use_sdf:
            # For SDF, extract at level 0
            level = 0.0
        else:
            # For occupancy, use threshold
            level = threshold
        
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                voxels,
                level=level,
                spacing=(self.voxel_size, self.voxel_size, self.voxel_size)
            )
        except Exception as e:
            print(f"Warning: Marching cubes failed ({e}), returning empty mesh")
            verts = np.zeros((0, 3))
            faces = np.zeros((0, 3), dtype=int)
            normals = np.zeros((0, 3))
        
        return {
            'vertices': verts,
            'faces': faces,
            'normals': normals
        }
    
    def save_mesh(self, mesh: Dict, output_path: str, format: str = 'ply'):
        """
        Save mesh to file
        
        Args:
            mesh: Mesh dictionary
            output_path: Output file path
            format: File format ('ply', 'obj', 'stl')
        """
        if len(mesh['vertices']) == 0:
            print("Warning: Empty mesh, saving placeholder")
            # Create a small cube as placeholder
            mesh = self.create_placeholder_mesh()
        
        # Create trimesh object
        mesh_obj = trimesh.Trimesh(
            vertices=mesh['vertices'],
            faces=mesh['faces'],
            vertex_normals=mesh['normals']
        )
        
        # Save
        mesh_obj.export(output_path)
    
    def create_placeholder_mesh(self):
        """Create a simple cube mesh"""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ]) * self.voxel_size
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5]
        ])
        
        normals = np.zeros_like(verts)
        
        return {'vertices': verts, 'faces': faces, 'normals': normals}
    
    def voxels_to_point_cloud(self, voxels: np.ndarray, threshold: float = 0.5):
        """
        Convert voxel grid to point cloud
        
        Args:
            voxels: (R, R, R) voxel grid
            threshold: Occupancy threshold
        Returns:
            points: (N, 3) point cloud
        """
        # Get occupied voxels
        occupied = voxels > threshold
        indices = np.argwhere(occupied)
        
        # Convert to coordinates
        points = indices * self.voxel_size
        
        return points


class Visualizer:
    """
    Visualization utilities
    """
    def __init__(self):
        pass
    
    def show_voxels(self, voxels: np.ndarray, threshold: float = 0.5):
        """
        Visualize voxel grid
        
        Args:
            voxels: (R, R, R) voxel grid
            threshold: Occupancy threshold
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get occupied voxels
        occupied = voxels > threshold
        
        # Plot
        ax.voxels(occupied, facecolors='blue', edgecolor='k', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Voxel Grid (occupancy > {threshold})')
        
        plt.show()
    
    def show_comparison(self, pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5):
        """
        Show prediction vs ground truth
        
        Args:
            pred: (R, R, R) predicted voxels
            gt: (R, R, R) ground truth voxels
            threshold: Occupancy threshold
        """
        fig = plt.figure(figsize=(15, 7))
        
        # Prediction
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.voxels(pred > threshold, facecolors='blue', edgecolor='k', alpha=0.5)
        ax1.set_title('Prediction')
        
        # Ground truth
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.voxels(gt > threshold, facecolors='green', edgecolor='k', alpha=0.5)
        ax2.set_title('Ground Truth')
        
        plt.show()
    
    def show_slices(self, voxels: np.ndarray, num_slices: int = 5):
        """
        Show slices of voxel grid
        
        Args:
            voxels: (R, R, R) voxel grid
            num_slices: Number of slices to show
        """
        R = voxels.shape[0]
        indices = np.linspace(0, R - 1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
        
        for i, idx in enumerate(indices):
            axes[i].imshow(voxels[:, :, idx], cmap='viridis')
            axes[i].set_title(f'Slice {idx}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


class MetricsCalculator:
    """
    Calculate evaluation metrics
    """
    @staticmethod
    def iou_3d(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate 3D IoU
        
        Args:
            pred: (R, R, R) predicted occupancy
            gt: (R, R, R) ground truth occupancy
            threshold: Binarization threshold
        Returns:
            iou: Intersection over Union
        """
        pred_binary = pred > threshold
        gt_binary = gt > threshold
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def chamfer_distance(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
        """
        Calculate Chamfer distance between point clouds
        
        Args:
            pred_points: (N, 3) predicted points
            gt_points: (M, 3) ground truth points
        Returns:
            chamfer: Chamfer distance
        """
        from scipy.spatial import cKDTree
        
        # Build KD-trees
        tree_pred = cKDTree(pred_points)
        tree_gt = cKDTree(gt_points)
        
        # Pred to GT
        dist_pred_to_gt, _ = tree_gt.query(pred_points)
        
        # GT to pred
        dist_gt_to_pred, _ = tree_pred.query(gt_points)
        
        # Chamfer distance
        chamfer = dist_pred_to_gt.mean() + dist_gt_to_pred.mean()
        
        return chamfer
    
    @staticmethod
    def precision_recall(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5):
        """
        Calculate precision and recall
        
        Args:
            pred: (R, R, R) predicted occupancy
            gt: (R, R, R) ground truth occupancy
            threshold: Binarization threshold
        Returns:
            precision, recall
        """
        pred_binary = pred > threshold
        gt_binary = gt > threshold
        
        tp = np.logical_and(pred_binary, gt_binary).sum()
        fp = np.logical_and(pred_binary, ~gt_binary).sum()
        fn = np.logical_and(~pred_binary, gt_binary).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall


def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate model on dataset
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: Device to use
    Returns:
        metrics: Dictionary of metrics
    """
    model.eval()
    
    metrics_calc = MetricsCalculator()
    
    total_iou = 0
    total_precision = 0
    total_recall = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            
            # Predict
            pred_occupancy, pred_sdf = model(video)
            
            if 'gt_voxels' in batch:
                gt = batch['gt_voxels'].numpy()
                pred = pred_occupancy.cpu().numpy()
                
                # Calculate metrics for each sample in batch
                for i in range(len(pred)):
                    iou = metrics_calc.iou_3d(pred[i], gt[i])
                    precision, recall = metrics_calc.precision_recall(pred[i], gt[i])
                    
                    total_iou += iou
                    total_precision += precision
                    total_recall += recall
                    num_samples += 1
    
    if num_samples == 0:
        return {}
    
    metrics = {
        'iou': total_iou / num_samples,
        'precision': total_precision / num_samples,
        'recall': total_recall / num_samples
    }
    
    return metrics