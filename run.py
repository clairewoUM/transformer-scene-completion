#!/usr/bin/env python
"""
run.py - Master script for all scene completion tasks

This is a unified interface for data preparation, training, inference, and visualization.

Usage:
    # Quick test (synthetic data)
    python run.py test
    
    # Prepare RealSense data
    python run.py prepare --input camera/colorFrames --output data/videos
    
    # Train model
    python run.py train --config config.yaml
    python run.py train --data data/videos --epochs 100  # Quick train
    
    # Run inference
    python run.py infer --video test.mp4 --checkpoint outputs/best.pt
    
    # Visualize results
    python run.py visualize --mesh scene.ply
    
    # Complete pipeline
    python run.py pipeline --input camera/colorFrames --output results/
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


class SceneCompletionRunner:
    """Master runner for all scene completion tasks"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
    
    def run_command(self, cmd):
        """Run shell command"""
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.project_root)
        if result.returncode != 0:
            print(f"Command failed with code {result.returncode}")
            sys.exit(1)
    
    def quick_test(self, args):
        """Quick test with synthetic data"""
        print("=" * 60)
        print("QUICK TEST MODE")
        print("=" * 60)
        print("This will train a small model on synthetic data")
        print("to verify the installation is working correctly.")
        print()
        
        cmd = [
            sys.executable, 'quick_start.py',
            '--mode', 'train',
            '--epochs', str(args.epochs),
            '--num_samples', '50',
            '--voxel_resolution', '32',
            '--output_dir', 'test_outputs'
        ]
        self.run_command(cmd)
        
        print()
        print("=" * 60)
        print("TEST COMPLETE!")
        print("=" * 60)
        print("If this succeeded, your installation is working correctly.")
    
    def prepare_data(self, args):
        """Prepare RealSense data"""
        print("=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)
        
        cmd = [
            sys.executable, 'prepare_realsense.py',
            '--input', args.input,
            '--output', args.output
        ]
        
        if args.fps:
            cmd.extend(['--fps', str(args.fps)])
        if args.skip_frames:
            cmd.extend(['--skip_frames', str(args.skip_frames)])
        if args.batch:
            cmd.append('--batch')
        
        self.run_command(cmd)
    
    def train_model(self, args):
        """Train scene completion model"""
        print("=" * 60)
        print("TRAINING")
        print("=" * 60)
        
        if args.config:
            cmd = [sys.executable, 'train.py', '--config', args.config]
        else:
            cmd = [
                sys.executable, 'train.py',
                '--data_type', args.data_type,
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--model_type', args.model_type
            ]
            
            if args.data:
                cmd.extend(['--data_dir', args.data])
            if args.output:
                cmd.extend(['--output_dir', args.output])
            if args.wandb:
                cmd.append('--use_wandb')
        
        self.run_command(cmd)
    
    def run_inference(self, args):
        """Run inference"""
        print("=" * 60)
        print("INFERENCE")
        print("=" * 60)
        
        cmd = [
            sys.executable, 'inference.py',
            '--checkpoint', args.checkpoint
        ]
        
        if args.video:
            cmd.extend(['--video', args.video])
        if args.video_dir:
            cmd.extend(['--video_dir', args.video_dir])
        if args.output:
            cmd.extend(['--output', args.output])
        if args.output_dir:
            cmd.extend(['--output_dir', args.output_dir])
        if args.webcam:
            cmd.append('--webcam')
        if args.display:
            cmd.append('--display')
        
        self.run_command(cmd)
    
    def visualize(self, args):
        """Visualize results"""
        print("=" * 60)
        print("VISUALIZATION")
        print("=" * 60)
        
        cmd = [sys.executable, 'visualize.py']
        
        if args.mesh:
            cmd.extend(['--mesh', args.mesh])
        if args.pred and args.gt:
            cmd.extend(['--pred', args.pred, '--gt', args.gt])
        if args.voxels:
            cmd.extend(['--voxels', args.voxels])
        if args.interactive:
            cmd.extend(['--interactive', '--checkpoint', args.checkpoint, '--video', args.video])
        
        self.run_command(cmd)
    
    def run_pipeline(self, args):
        """Complete pipeline from data to results"""
        print("=" * 60)
        print("COMPLETE PIPELINE")
        print("=" * 60)
        print()
        print("This will run the complete pipeline:")
        print("1. Prepare data")
        print("2. Train model")
        print("3. Run inference")
        print("4. Visualize results")
        print()
        
        # Step 1: Prepare data
        if args.input:
            print("\n[1/4] Preparing data...")
            prepare_args = argparse.Namespace(
                input=args.input,
                output='data/videos',
                fps=30,
                skip_frames=350,
                batch=False
            )
            self.prepare_data(prepare_args)
        
        # Step 2: Train model
        print("\n[2/4] Training model...")
        train_args = argparse.Namespace(
            config=args.config if hasattr(args, 'config') else None,
            data_type='video',
            data='data/videos',
            epochs=args.epochs if hasattr(args, 'epochs') else 100,
            batch_size=4,
            model_type='lightweight',
            output='outputs',
            wandb=False
        )
        self.train_model(train_args)
        
        # Find best checkpoint
        checkpoints = list(Path('outputs').glob('*/best.pt'))
        if not checkpoints:
            print("No checkpoint found!")
            return
        checkpoint = sorted(checkpoints)[-1]
        
        # Step 3: Run inference
        print("\n[3/4] Running inference...")
        infer_args = argparse.Namespace(
            checkpoint=str(checkpoint),
            video=args.video if hasattr(args, 'video') else None,
            video_dir='data/videos' if not hasattr(args, 'video') else None,
            output=None,
            output_dir=args.output if hasattr(args, 'output') else 'results',
            webcam=False,
            display=False
        )
        self.run_inference(infer_args)
        
        # Step 4: Visualize
        print("\n[4/4] Visualization...")
        print("Results saved to:", args.output if hasattr(args, 'output') else 'results')
        print("To view results, run:")
        print(f"  python visualize.py --mesh results/*.ply")
        
        print()
        print("=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)


def main():
    runner = SceneCompletionRunner()
    
    parser = argparse.ArgumentParser(
        description='Scene Completion Master Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python run.py test
  
  # Prepare RealSense data
  python run.py prepare --input camera/colorFrames --output data/videos
  
  # Train model
  python run.py train --data data/videos --epochs 100
  
  # Run inference
  python run.py infer --video test.mp4 --checkpoint outputs/best.pt --output scene.ply
  
  # Visualize
  python run.py visualize --mesh scene.ply
  
  # Complete pipeline
  python run.py pipeline --input camera/colorFrames --output results/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Quick test with synthetic data')
    test_parser.add_argument('--epochs', type=int, default=10)
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare RealSense data')
    prepare_parser.add_argument('--input', type=str, required=True)
    prepare_parser.add_argument('--output', type=str, required=True)
    prepare_parser.add_argument('--fps', type=int, default=30)
    prepare_parser.add_argument('--skip_frames', type=int, default=350)
    prepare_parser.add_argument('--batch', action='store_true')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', type=str)
    train_parser.add_argument('--data', type=str)
    train_parser.add_argument('--data_type', type=str, default='video')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch_size', type=int, default=4)
    train_parser.add_argument('--model_type', type=str, default='lightweight')
    train_parser.add_argument('--output', type=str, default='outputs')
    train_parser.add_argument('--wandb', action='store_true')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str, required=True)
    infer_parser.add_argument('--video', type=str)
    infer_parser.add_argument('--video_dir', type=str)
    infer_parser.add_argument('--output', type=str)
    infer_parser.add_argument('--output_dir', type=str)
    infer_parser.add_argument('--webcam', action='store_true')
    infer_parser.add_argument('--display', action='store_true')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize results')
    viz_parser.add_argument('--mesh', type=str)
    viz_parser.add_argument('--pred', type=str)
    viz_parser.add_argument('--gt', type=str)
    viz_parser.add_argument('--voxels', type=str)
    viz_parser.add_argument('--interactive', action='store_true')
    viz_parser.add_argument('--checkpoint', type=str)
    viz_parser.add_argument('--video', type=str)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--input', type=str)
    pipeline_parser.add_argument('--output', type=str, default='results')
    pipeline_parser.add_argument('--config', type=str)
    pipeline_parser.add_argument('--epochs', type=int, default=100)
    pipeline_parser.add_argument('--video', type=str)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run command
    if args.command == 'test':
        runner.quick_test(args)
    elif args.command == 'prepare':
        runner.prepare_data(args)
    elif args.command == 'train':
        runner.train_model(args)
    elif args.command == 'infer':
        runner.run_inference(args)
    elif args.command == 'visualize':
        runner.visualize(args)
    elif args.command == 'pipeline':
        runner.run_pipeline(args)


if __name__ == '__main__':
    main()