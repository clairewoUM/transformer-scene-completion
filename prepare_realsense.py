"""
prepare_realsense.py - Prepare RealSense D455 data for training

This script converts your RealSense frame sequences into videos
and prepares them for the scene completion model.file:///home/dudl3443/transformer-scene/train.py


Usage:
    python prepare_realsense.py --input colorFrames --output data/videos
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import re  

def natural_sort_key(s):
    """Sort numbers correctly: 1, 2, 3... 10, 11 not 1, 10, 11... 2"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', str(s))]

class RealSenseDataPreparator:
    """Prepare RealSense data for scene completion training"""
    
    def __init__(self, fps=30, output_resolution=(640, 480)):
        self.fps = fps
        self.output_resolution = output_resolution
    

    def frames_to_video(self, frames_dir, output_video, skip_frames=0):
        """
        Convert frame sequence to video
        
        Args:
            frames_dir: Directory with frame_*.png files
            output_video: Output video path
            skip_frames: Skip first N frames (e.g., stationary period)
        """
        frames_dir = Path(frames_dir)
        
        # Find all frames
        frames = sorted(frames_dir.glob('frame_*.png'), key=natural_sort_key)
        
        if not frames:
            print(f"No frames found in {frames_dir}")
            return False
        
        print(f"Found {len(frames)} frames")
        
        # Skip initial frames
        frames = frames[skip_frames:]
        print(f"Processing {len(frames)} frames (skipped first {skip_frames})")
        
        if not frames:
            print("No frames left after skipping!")
            return False
        
        # Get frame size
        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            print(f"Could not read first frame: {frames[0]}")
            return False
        
        h, w = first_frame.shape[:2]
        
        # Resize if needed
        if (w, h) != self.output_resolution:
            print(f"Resizing from {w}x{h} to {self.output_resolution}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_video),
            fourcc,
            self.fps,
            self.output_resolution
        )
        
        # Write frames
        for frame_path in tqdm(frames, desc="Writing video"):
            frame = cv2.imread(str(frame_path))
            
            if frame is not None:
                if (frame.shape[1], frame.shape[0]) != self.output_resolution:
                    frame = cv2.resize(frame, self.output_resolution)
                out.write(frame)
        
        out.release()
        print(f"Created video: {output_video}")
        
        return True
    
    def prepare_depth_video(self, depth_dir, output_video, skip_frames=0):
        """
        Convert depth frames to video (visualized)
        
        Args:
            depth_dir: Directory with depth frame_*.png files
            output_video: Output video path
            skip_frames: Skip first N frames
        """
        depth_dir = Path(depth_dir)
        
        # Find all depth frames
        frames = sorted(depth_dir.glob('frame_*.png'), key=natural_sort_key)
        frames = frames[skip_frames:]
        
        if not frames:
            print(f"No depth frames found in {depth_dir}")
            return False
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_video),
            fourcc,
            self.fps,
            self.output_resolution
        )
        
        for frame_path in tqdm(frames, desc="Writing depth video"):
            # Read depth (16-bit)
            depth = cv2.imread(str(frame_path), cv2.IMREAD_ANYDEPTH)
            
            if depth is not None:
                # Normalize for visualization
                depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
                
                # Resize
                if (depth_color.shape[1], depth_color.shape[0]) != self.output_resolution:
                    depth_color = cv2.resize(depth_color, self.output_resolution)
                
                out.write(depth_color)
        
        out.release()
        print(f"Created depth video: {output_video}")
        
        return True
    
    def create_dataset_split(self, video_dir, train_ratio=0.9):
        """
        Create train/val split
        
        Args:
            video_dir: Directory with video files
            train_ratio: Ratio of training data
        """
        video_dir = Path(video_dir)
        
        # Find all videos
        videos = list(video_dir.glob('*.mp4'))
        videos.extend(list(video_dir.glob('*.avi')))
        
        if not videos:
            print(f"No videos found in {video_dir}")
            return
        
        # Shuffle
        np.random.shuffle(videos)
        
        # Split
        split_idx = int(len(videos) * train_ratio)
        train_videos = videos[:split_idx]
        val_videos = videos[split_idx:]
        
        # Save lists
        with open(video_dir / 'train_list.txt', 'w') as f:
            for v in train_videos:
                f.write(f"{v.name}\n")
        
        with open(video_dir / 'val_list.txt', 'w') as f:
            for v in val_videos:
                f.write(f"{v.name}\n")
        
        print(f"Created split: {len(train_videos)} train, {len(val_videos)} val")
    
    def process_realsense_recording(self, 
                                   input_dir,
                                   output_dir,
                                   skip_stationary_frames=350):
        """
        Process complete RealSense recording
        
        Args:
            input_dir: Directory with /colorFrames and /depthFrames
            output_dir: Output directory for processed videos
            skip_stationary_frames: Number of initial stationary frames to skip
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Processing RealSense Recording")
        print("=" * 60)
        
        # Process color frames
        color_dir = input_dir / 'colorFrames'
        if color_dir.exists():
            print("\nProcessing color frames...")
            self.frames_to_video(
                color_dir,
                output_dir / 'color.mp4',
                skip_frames=skip_stationary_frames
            )
        else:
            print(f"Color directory not found: {color_dir}")
        
        # Process depth frames
        depth_dir = input_dir / 'depthFrames'
        if depth_dir.exists():
            print("\nProcessing depth frames...")
            self.prepare_depth_video(
                depth_dir,
                output_dir / 'depth_vis.mp4',
                skip_frames=skip_stationary_frames
            )
        else:
            print(f"Depth directory not found: {depth_dir}")
        
        # Create metadata
        metadata = {
            'source': str(input_dir),
            'fps': self.fps,
            'resolution': self.output_resolution,
            'skip_frames': skip_stationary_frames,
            'has_color': color_dir.exists(),
            'has_depth': depth_dir.exists()
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nProcessing complete! Output: {output_dir}")
    
    def batch_process(self, input_root, output_root, pattern='*/camera'):
        """
        Batch process multiple recordings
        
        Args:
            input_root: Root directory with multiple recordings
            output_root: Output root directory
            pattern: Pattern to find camera directories
        """
        input_root = Path(input_root)
        output_root = Path(output_root)
        
        # Find all camera directories
        camera_dirs = sorted(input_root.glob(pattern))
        
        print(f"Found {len(camera_dirs)} recordings")
        
        for i, camera_dir in enumerate(camera_dirs):
            recording_name = camera_dir.parent.name
            print(f"\n[{i+1}/{len(camera_dirs)}] Processing: {recording_name}")
            
            output_dir = output_root / recording_name
            
            self.process_realsense_recording(
                camera_dir.parent,
                output_dir
            )
        
        # Create dataset split
        print("\nCreating train/val split...")
        self.create_dataset_split(output_root)
        
        print("\nBatch processing complete!")


def main():
    parser = argparse.ArgumentParser(description='Prepare RealSense data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory (containing colorFrames)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output video FPS')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Output resolution (WxH)')
    parser.add_argument('--skip_frames', type=int, default=350,
                       help='Skip first N frames (stationary period)')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process multiple recordings')
    
    args = parser.parse_args()
    
    # Parse resolution
    w, h = map(int, args.resolution.split('x'))
    
    # Create preparator
    preparator = RealSenseDataPreparator(
        fps=args.fps,
        output_resolution=(w, h)
    )
    
    if args.batch:
        # Batch processing
        preparator.batch_process(
            args.input,
            args.output
        )
    else:
        # Single recording
        preparator.process_realsense_recording(
            args.input,
            args.output,
            skip_stationary_frames=args.skip_frames
        )


if __name__ == '__main__':
    main()