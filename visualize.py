"""
visualize.py - Visualization tools for scene completion

Usage:
    # View 3D mesh
    python visualize.py --mesh scene.ply
    
    # Compare prediction and ground truth
    python visualize.py --pred pred.ply --gt gt.ply
    
    # View voxel grid
    python visualize.py --voxels voxels.npz
    
    # Interactive viewer
    python visualize.py --interactive --checkpoint model.pt --video test.mp4
"""

import numpy as np
import argparse
from pathlib import Path
import torch
import cv2

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed. 3D visualization will be limited.")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class SceneVisualizer:
    """Visualization tools for 3D scene completion"""
    
    def __init__(self):
        self.use_open3d = HAS_OPEN3D
    
    def view_mesh(self, mesh_path, show_normals=False):
        """
        View 3D mesh
        
        Args:
            mesh_path: Path to mesh file (.ply, .obj, .stl)
            show_normals: Show surface normals
        """
        mesh_path = Path(mesh_path)
        
        if not mesh_path.exists():
            print(f"Mesh not found: {mesh_path}")
            return
        
        if self.use_open3d:
            print(f"Loading mesh: {mesh_path}")
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            
            # Compute normals if not present
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            # Visualize
            geometries = [mesh]
            
            if show_normals:
                # Create coordinate frame
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5, origin=[0, 0, 0]
                )
                geometries.append(coord_frame)
            
            print("Controls:")
            print("  - Mouse: Rotate")
            print("  - Scroll: Zoom")
            print("  - Shift+Mouse: Pan")
            
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"Mesh Viewer: {mesh_path.name}",
                width=1024,
                height=768
            )
        else:
            print("Install Open3D for 3D visualization: pip install open3d")
    
    def compare_meshes(self, pred_path, gt_path):
        """
        Compare prediction and ground truth meshes
        
        Args:
            pred_path: Path to predicted mesh
            gt_path: Path to ground truth mesh
        """
        if not self.use_open3d:
            print("Install Open3D for mesh comparison")
            return
        
        print("Loading meshes...")
        pred_mesh = o3d.io.read_triangle_mesh(str(pred_path))
        gt_mesh = o3d.io.read_triangle_mesh(str(gt_path))
        
        # Color meshes differently
        pred_mesh.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
        gt_mesh.paint_uniform_color([0.0, 1.0, 0.0])    # Green
        
        # Compute normals
        pred_mesh.compute_vertex_normals()
        gt_mesh.compute_vertex_normals()
        
        print("Controls:")
        print("  Blue: Prediction")
        print("  Green: Ground Truth")
        
        o3d.visualization.draw_geometries(
            [pred_mesh, gt_mesh],
            window_name="Comparison: Prediction (Blue) vs Ground Truth (Green)",
            width=1024,
            height=768
        )
    
    def view_voxels(self, voxels, threshold=0.5):
        """
        Visualize voxel grid
        
        Args:
            voxels: (R, R, R) voxel grid or path to .npz file
            threshold: Occupancy threshold
        """
        # Load voxels if path
        if isinstance(voxels, (str, Path)):
            data = np.load(voxels)
            voxels = data['voxels'] if 'voxels' in data else data['occupancy']
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D view
        ax1 = fig.add_subplot(131, projection='3d')
        occupied = voxels > threshold
        ax1.voxels(occupied, facecolors='blue', edgecolor='k', alpha=0.5)
        ax1.set_title(f'3D View (threshold={threshold})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Slice views
        mid_z = voxels.shape[2] // 2
        ax2 = fig.add_subplot(132)
        ax2.imshow(voxels[:, :, mid_z], cmap='viridis')
        ax2.set_title(f'XY Slice (z={mid_z})')
        ax2.axis('off')
        
        mid_x = voxels.shape[0] // 2
        ax3 = fig.add_subplot(133)
        ax3.imshow(voxels[mid_x, :, :], cmap='viridis')
        ax3.set_title(f'YZ Slice (x={mid_x})')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def view_video_reconstruction(self, video_path, voxels):
        """
        Show video alongside reconstruction
        
        Args:
            video_path: Path to input video
            voxels: Reconstructed voxel grid
        """
        cap = cv2.VideoCapture(str(video_path))
        
        fig = plt.figure(figsize=(15, 5))
        
        # Video frame
        ax1 = fig.add_subplot(121)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_plot = ax1.imshow(frame)
        ax1.set_title('Input Video')
        ax1.axis('off')
        
        # Voxels
        ax2 = fig.add_subplot(122, projection='3d')
        occupied = voxels > 0.5
        ax2.voxels(occupied, facecolors='blue', edgecolor='k', alpha=0.5)
        ax2.set_title('Reconstruction')
        
        def update(frame_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_plot.set_array(frame)
            return [img_plot]
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        anim = FuncAnimation(
            fig, update,
            frames=range(0, total_frames, 5),
            interval=100,
            blit=True
        )
        
        plt.tight_layout()
        plt.show()
        
        cap.release()
    
    def interactive_viewer(self, checkpoint_path, video_path):
        """
        Interactive viewer with live reconstruction
        
        Args:
            checkpoint_path: Path to model checkpoint
            video_path: Path to input video
        """
        print("Loading model...")
        from model import LightweightSceneCompletion
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = LightweightSceneCompletion(
            voxel_resolution=checkpoint['config']['voxel_resolution']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load video
        from utils import VideoProcessor
        processor = VideoProcessor(
            img_size=checkpoint['config']['img_size'],
            num_frames=checkpoint['config']['num_frames']
        )
        
        video = processor.load_video(video_path)
        
        # Run inference
        print("Running reconstruction...")
        with torch.no_grad():
            voxels = model(video.unsqueeze(0).to(device))
        
        voxels = voxels.cpu().numpy()[0]
        
        # Visualize
        self.view_video_reconstruction(video_path, voxels)
    
    def create_turntable_animation(self, mesh_path, output_path='turntable.gif', num_frames=36):
        """
        Create turntable animation of mesh
        
        Args:
            mesh_path: Path to mesh file
            output_path: Output GIF path
            num_frames: Number of frames in animation
        """
        if not self.use_open3d:
            print("Install Open3D for animation")
            return
        
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(mesh)
        
        frames = []
        for i in range(num_frames):
            # Rotate view
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            vis.poll_events()
            vis.update_renderer()
            
            # Capture frame
            frame = np.asarray(vis.capture_screen_float_buffer())
            frames.append((frame * 255).astype(np.uint8))
        
        vis.destroy_window()
        
        # Save as GIF
        import imageio
        imageio.mimsave(output_path, frames, fps=10)
        print(f"Saved animation to {output_path}")
    
    def plot_training_progress(self, log_file):
        """
        Plot training curves from log file
        
        Args:
            log_file: Path to training log (JSON)
        """
        import json
        
        with open(log_file, 'r') as f:
            logs = [json.loads(line) for line in f]
        
        epochs = [log['epoch'] for log in logs]
        train_loss = [log['train_loss'] for log in logs]
        val_loss = [log['val_loss'] for log in logs]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(epochs, train_loss, label='Train Loss', marker='o')
        ax.plot(epochs, val_loss, label='Val Loss', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize scene completion results')
    parser.add_argument('--mesh', type=str, help='Path to mesh file')
    parser.add_argument('--pred', type=str, help='Path to predicted mesh')
    parser.add_argument('--gt', type=str, help='Path to ground truth mesh')
    parser.add_argument('--voxels', type=str, help='Path to voxel file (.npz)')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--interactive', action='store_true', help='Interactive viewer')
    parser.add_argument('--normals', action='store_true', help='Show normals')
    parser.add_argument('--threshold', type=float, default=0.5, help='Voxel threshold')
    parser.add_argument('--animation', type=str, help='Create turntable animation')
    parser.add_argument('--log', type=str, help='Plot training log')
    
    args = parser.parse_args()
    
    visualizer = SceneVisualizer()
    
    if args.interactive and args.checkpoint and args.video:
        visualizer.interactive_viewer(args.checkpoint, args.video)
    
    elif args.pred and args.gt:
        visualizer.compare_meshes(args.pred, args.gt)
    
    elif args.mesh:
        if args.animation:
            visualizer.create_turntable_animation(args.mesh, args.animation)
        else:
            visualizer.view_mesh(args.mesh, show_normals=args.normals)
    
    elif args.voxels:
        visualizer.view_voxels(args.voxels, threshold=args.threshold)
    
    elif args.log:
        visualizer.plot_training_progress(args.log)
    
    else:
        print("Error: Please specify visualization mode")
        print("Examples:")
        print("  python visualize.py --mesh scene.ply")
        print("  python visualize.py --pred pred.ply --gt gt.ply")
        print("  python visualize.py --voxels voxels.npz")
        print("  python visualize.py --interactive --checkpoint model.pt --video test.mp4")


if __name__ == '__main__':
    main()