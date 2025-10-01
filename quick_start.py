"""
quick_start.py - All-in-one script for immediate testing

This combines all necessary components in a single file for easy testing.
For production use, use the modular version (model.py, train.py, etc.)

Usage:
    # Train on synthetic data
    python quick_start.py --mode train --epochs 10
    
    # Run inference
    python quick_start.py --mode infer --video test.mp4 --output scene.ply
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# SIMPLIFIED MODEL
# ============================================================================

class QuickSceneCompletion(nn.Module):
    """Lightweight scene completion model"""
    def __init__(self, voxel_resolution=64):
        super().__init__()
        self.voxel_resolution = voxel_resolution
        
        # Video encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Temporal aggregation
        self.temporal_gru = nn.GRU(256 * 64, 512, 2, batch_first=True)
        
        # 3D decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, voxel_resolution ** 3)
        )
    
    def forward(self, video):
        B, T, C, H, W = video.shape
        
        # Encode frames
        video_flat = video.view(B * T, C, H, W)
        features = self.encoder(video_flat)
        features = features.view(B, T, -1)
        
        # Temporal aggregation
        _, hidden = self.temporal_gru(features)
        features = hidden[-1]
        
        # Decode to 3D
        voxels = self.decoder(features)
        voxels = voxels.view(B, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution)
        voxels = torch.sigmoid(voxels)
        
        return voxels


# ============================================================================
# DATA LOADING
# ============================================================================

def load_video(video_path, num_frames=10, img_size=224):
    """Load and preprocess video"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Could not read video: {video_path}")
    
    # Sample frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
    
    cap.release()
    
    # To tensor
    frames = np.stack(frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    
    return frames


def create_synthetic_data(num_samples=100, num_frames=10, img_size=224, voxel_resolution=64):
    """Create synthetic data for testing"""
    data = []
    
    for _ in range(num_samples):
        # Random video
        video = torch.randn(num_frames, 3, img_size, img_size)
        
        # Random voxel grid with structure
        x = torch.linspace(-1, 1, voxel_resolution)
        y = torch.linspace(-1, 1, voxel_resolution)
        z = torch.linspace(-1, 1, voxel_resolution)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Create sphere
        radius = 0.3 + 0.2 * torch.rand(1).item()
        voxels = ((X**2 + Y**2 + Z**2) < radius**2).float()
        voxels += 0.05 * torch.randn_like(voxels)
        voxels = torch.clamp(voxels, 0, 1)
        
        data.append({'video': video, 'voxels': voxels})
    
    return data


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    """Training loop"""
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = QuickSceneCompletion(voxel_resolution=args.voxel_resolution).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create data
    print("Creating synthetic data...")
    train_data = create_synthetic_data(
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        img_size=args.img_size,
        voxel_resolution=args.voxel_resolution
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_data, desc=f'Epoch {epoch}')
        for sample in pbar:
            video = sample['video'].unsqueeze(0).to(device)
            gt_voxels = sample['voxels'].unsqueeze(0).to(device)
            
            # Forward
            pred_voxels = model(video)
            
            # Loss
            loss = F.binary_cross_entropy(pred_voxels, gt_voxels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    # Save model
    output_path = Path(args.output_dir) / 'model.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'voxel_resolution': args.voxel_resolution,
            'img_size': args.img_size,
            'num_frames': args.num_frames
        }
    }, output_path)
    
    print(f"\nModel saved to {output_path}")


# ============================================================================
# INFERENCE
# ============================================================================

def extract_mesh(voxels, threshold=0.5, voxel_size=0.05):
    """Extract mesh from voxels using marching cubes"""
    from skimage import measure
    
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            voxels,
            level=threshold,
            spacing=(voxel_size, voxel_size, voxel_size)
        )
    except:
        print("Warning: Marching cubes failed, creating empty mesh")
        verts = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=int)
        normals = np.zeros((0, 3))
    
    return verts, faces, normals


def save_mesh_ply(vertices, faces, normals, output_path):
    """Save mesh as PLY file"""
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Vertices
        for v, n in zip(vertices, normals):
            f.write(f"{v[0]} {v[1]} {v[2]} {n[0]} {n[1]} {n[2]}\n")
        
        # Faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def infer(args):
    """Inference"""
    print("=" * 60)
    print("INFERENCE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    model = QuickSceneCompletion(
        voxel_resolution=config['voxel_resolution']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {args.checkpoint}")
    
    # Load video
    print(f"Loading video: {args.video}")
    video = load_video(
        args.video,
        num_frames=config['num_frames'],
        img_size=config['img_size']
    )
    video = video.unsqueeze(0).to(device)
    
    # Run inference
    print("Running scene completion...")
    with torch.no_grad():
        voxels = model(video)
    
    voxels = voxels.cpu().numpy()[0]
    
    print(f"Generated voxel grid: {voxels.shape}")
    print(f"Occupancy: {voxels.mean():.3f}")
    
    # Extract mesh
    print("Extracting mesh...")
    vertices, faces, normals = extract_mesh(voxels, threshold=0.5)
    
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Save
    print(f"Saving to {args.output}")
    save_mesh_ply(vertices, faces, normals, args.output)
    
    print("Done!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Quick start scene completion')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'])
    
    # Training args
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    
    # Inference args
    parser.add_argument('--checkpoint', type=str, default='outputs/model.pt')
    parser.add_argument('--video', type=str, help='Input video path')
    parser.add_argument('--output', type=str, default='scene.ply')
    
    # Common args
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--num_frames', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--voxel_resolution', type=int, default=64)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        if not args.video:
            print("Error: --video required for inference")
            return
        infer(args)


if __name__ == '__main__':
    main()