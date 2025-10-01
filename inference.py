"""
inference.py - Inference script for scene completion

Usage:
    # Single video
    python inference.py --video path/to/video.mp4 --checkpoint outputs/best.pt --output scene.ply
    
    # Batch processing
    python inference.py --video_dir path/to/videos/ --checkpoint outputs/best.pt --output_dir results/
    
    # Real-time from webcam
    python inference.py --webcam --checkpoint outputs/best.pt --display
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json

from model import PoseFreeSceneCompletion, LightweightSceneCompletion
from utils import VideoProcessor, MeshExporter, Visualizer


class SceneCompletionInference:
    """
    Inference pipeline for scene completion
    """
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model type: {self.config.get('model_type', 'lightweight')}")
        
        # Initialize utilities
        self.video_processor = VideoProcessor(
            img_size=self.config['img_size'],
            num_frames=self.config['num_frames']
        )
        self.mesh_exporter = MeshExporter(voxel_size=0.05)
    
    def create_model(self):
        """Create model from config"""
        model_type = self.config.get('model_type', 'lightweight')
        
        if model_type == 'lightweight':
            model = LightweightSceneCompletion(
                voxel_resolution=self.config['voxel_resolution']
            )
        else:
            model = PoseFreeSceneCompletion(
                img_size=self.config['img_size'],
                voxel_resolution=self.config['voxel_resolution'],
                embed_dim=self.config.get('embed_dim', 512),
                encoder_depth=self.config.get('encoder_depth', 8),
                decoder_depth=self.config.get('decoder_depth', 4),
                num_heads=self.config.get('num_heads', 8)
            )
        
        return model
    
    def process_video(self, video_path, output_path=None):
        """
        Process a single video
        
        Args:
            video_path: Path to video file
            output_path: Path to save output mesh
        Returns:
            occupancy: (R, R, R) occupancy grid
            mesh: Mesh dictionary
        """
        print(f"Processing video: {video_path}")
        
        # Load and preprocess video
        video = self.video_processor.load_video(video_path)
        video = video.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Run inference
        print("Running scene completion...")
        with torch.no_grad():
            occupancy, sdf = self.model(video)
        
        occupancy = occupancy.cpu().numpy()[0]
        sdf = sdf.cpu().numpy()[0]
        
        print(f"Generated scene: {occupancy.shape}")
        print(f"Occupancy range: [{occupancy.min():.3f}, {occupancy.max():.3f}]")
        print(f"Mean occupancy: {occupancy.mean():.3f}")
        
        # Extract mesh
        print("Extracting mesh...")
        mesh = self.mesh_exporter.extract_mesh(occupancy, threshold=0.5)
        
        print(f"Mesh: {len(mesh['vertices'])} vertices, {len(mesh['faces'])} faces")
        
        # Save mesh
        if output_path:
            self.mesh_exporter.save_mesh(mesh, output_path)
            print(f"Saved mesh to {output_path}")
        
        return occupancy, mesh
    
    def process_video_dir(self, video_dir, output_dir):
        """
        Process all videos in a directory
        
        Args:
            video_dir: Directory containing videos
            output_dir: Directory to save output meshes
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all videos
        videos = list(video_dir.glob('*.mp4'))
        videos.extend(list(video_dir.glob('*.avi')))
        
        print(f"Found {len(videos)} videos")
        
        # Process each video
        for video_path in tqdm(videos):
            output_path = output_dir / f"{video_path.stem}.ply"
            
            try:
                self.process_video(video_path, output_path)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
    
    def process_webcam(self, display=True, save_path=None):
        """
        Process real-time webcam stream
        
        Args:
            display: Whether to display visualization
            save_path: Path to save final mesh
        """
        cap = cv2.VideoCapture(0)
        
        frame_buffer = []
        
        print("Starting webcam capture...")
        print("Press 'q' to quit and save")
        print("Press 'space' to capture frame")
        
        visualizer = Visualizer() if display else None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display frame
                cv2.imshow('Webcam', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Capture frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_buffer.append(frame_rgb)
                    print(f"Captured frame {len(frame_buffer)}")
                    
                    # Process when we have enough frames
                    if len(frame_buffer) >= self.config['num_frames']:
                        print("Processing captured frames...")
                        
                        # Prepare video
                        video = self.video_processor.prepare_frames(frame_buffer[-self.config['num_frames']:])
                        video = video.unsqueeze(0).to(self.device)
                        
                        # Run inference
                        with torch.no_grad():
                            occupancy, sdf = self.model(video)
                        
                        occupancy = occupancy.cpu().numpy()[0]
                        
                        # Visualize
                        if visualizer:
                            visualizer.show_voxels(occupancy)
                        
                        print(f"Scene occupancy: {occupancy.mean():.3f}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Save final mesh if frames were captured
        if len(frame_buffer) >= self.config['num_frames'] and save_path:
            print("Generating final mesh...")
            video = self.video_processor.prepare_frames(frame_buffer[-self.config['num_frames']:])
            video = video.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                occupancy, sdf = self.model(video)
            
            occupancy = occupancy.cpu().numpy()[0]
            mesh = self.mesh_exporter.extract_mesh(occupancy, threshold=0.5)
            self.mesh_exporter.save_mesh(mesh, save_path)
            print(f"Saved final mesh to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Scene completion inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--video_dir', type=str, help='Directory of videos')
    parser.add_argument('--output', type=str, help='Output mesh path')
    parser.add_argument('--output_dir', type=str, help='Output directory for batch processing')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--display', action='store_true', help='Display visualization')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = SceneCompletionInference(args.checkpoint, device=args.device)
    
    if args.webcam:
        # Real-time webcam
        inference.process_webcam(
            display=args.display,
            save_path=args.output or 'webcam_scene.ply'
        )
    
    elif args.video:
        # Single video
        inference.process_video(
            args.video,
            output_path=args.output or f"{Path(args.video).stem}_scene.ply"
        )
    
    elif args.video_dir:
        # Batch processing
        inference.process_video_dir(
            args.video_dir,
            output_dir=args.output_dir or 'results'
        )
    
    else:
        print("Error: Please specify --video, --video_dir, or --webcam")
        return


if __name__ == '__main__':
    main()