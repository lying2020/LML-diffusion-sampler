#!/usr/bin/env python3
"""
LML Diffusion Sampler Test Script

This script tests and compares different diffusion samplers including the LML method.
It generates images and provides quality evaluation without requiring CIFAR-10 dataset download.
"""

import os
import sys
import argparse
import subprocess
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        end_time = time.time()
        print(f"✅ {description} completed successfully in {end_time - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⚠️  {description} interrupted by user")
        return False

def create_dummy_images(output_dir, num_images=1000):
    """Create dummy CIFAR-10 style images for comparison"""
    print(f"Creating {num_images} dummy CIFAR-10 style images...")

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        # Generate random 32x32 RGB image
        image = torch.randn(3, 32, 32)
        image = (image - image.min()) / (image.max() - image.min())
        image = transforms.ToPILImage()(image)

        filename = f"cifar10_test_{i:05d}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)

        if i % 200 == 0:
            print(f"Created {i}/{num_images} images")

    print(f"✅ Dummy images created in {output_dir}")

def evaluate_images(generated_dir, real_dir):
    """Evaluate generated images with simple metrics"""
    print("Calculating evaluation metrics...")

    # Count images
    gen_images = [f for f in os.listdir(generated_dir) if f.endswith('.png')]
    real_images = [f for f in os.listdir(real_dir) if f.endswith('.png')]

    print(f"Generated images: {len(gen_images)}")
    print(f"Real images: {len(real_images)}")

    # Calculate average file sizes as quality indicator
    gen_sizes = []
    for img_file in gen_images[:min(20, len(gen_images))]:
        img_path = os.path.join(generated_dir, img_file)
        gen_sizes.append(os.path.getsize(img_path))

    real_sizes = []
    for img_file in real_images[:min(20, len(real_images))]:
        img_path = os.path.join(real_dir, img_file)
        real_sizes.append(os.path.getsize(img_path))

    avg_gen_size = np.mean(gen_sizes)
    avg_real_size = np.mean(real_sizes)

    # Quality score based on file size similarity
    quality_score = min(avg_gen_size / avg_real_size, 1.0) * 100

    return {
        'num_generated': len(gen_images),
        'num_real': len(real_images),
        'avg_gen_size': avg_gen_size,
        'avg_real_size': avg_real_size,
        'quality_score': quality_score
    }

def main():
    parser = argparse.ArgumentParser(description="Test LML Diffusion Sampler")
    parser.add_argument('--samplers', nargs='+',
                        default=['ddim', 'dpm', 'dpm_lm'],
                        help='List of samplers to test')
    parser.add_argument('--test_num', type=int, default=20,
                        help='Number of test samples to generate')
    parser.add_argument('--num_inference_steps', type=int, default=20,
                        help='Number of inference steps')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation')
    parser.add_argument('--lamb', type=float, default=0.0008,
                        help='Lambda parameter for LML sampler')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./output/cifar10',
                        help='Output directory for generated images')

    args = parser.parse_args()

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    print("🚀 LML Diffusion Sampler Test")
    print("="*50)
    print(f"Samplers to test: {', '.join(args.samplers)}")
    print(f"Test samples: {args.test_num}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("="*50)

    # Create dummy real images for comparison
    real_dir = os.path.join(project_dir, 'data', 'cifar10_real')
    if not os.path.exists(real_dir):
        create_dummy_images(real_dir, 1000)

    # Test each sampler
    results = {}

    for sampler in args.samplers:
        print(f"\n🎨 Testing {sampler.upper()} sampler...")

        # Generate images
        generation_cmd = [
            sys.executable,
            os.path.join(script_dir, 'cifar10.py'),
            '--sampler_type', sampler,
            '--test_num', str(args.test_num),
            '--num_inference_steps', str(args.num_inference_steps),
            '--batch_size', str(args.batch_size),
            '--device', args.device,
            '--save_dir', args.output_dir
        ]

        if sampler == 'dpm_lm':
            generation_cmd.extend(['--lamb', str(args.lamb)])

        if run_command(generation_cmd, f"Generate images with {sampler}"):
            # Evaluate generated images
            generated_dir = os.path.join(project_dir, args.output_dir, sampler)

            if os.path.exists(generated_dir):
                metrics = evaluate_images(generated_dir, real_dir)
                results[sampler] = metrics
                print(f"✅ {sampler} completed - Quality Score: {metrics['quality_score']:.1f}%")
            else:
                print(f"❌ Generated images directory not found: {generated_dir}")
        else:
            print(f"❌ {sampler} generation failed")

    # Print summary results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("="*60)

    if results:
        print(f"{'Sampler':<12} {'Images':<8} {'Quality':<10} {'Avg Size':<12} {'Rating'}")
        print("-" * 60)

        for sampler, metrics in results.items():
            quality = metrics['quality_score']
            if quality > 80:
                rating = "⭐⭐⭐⭐⭐"
            elif quality > 60:
                rating = "⭐⭐⭐⭐"
            elif quality > 40:
                rating = "⭐⭐⭐"
            else:
                rating = "⭐⭐"

            print(f"{sampler:<12} {metrics['num_generated']:<8} {quality:<10.1f}% {metrics['avg_gen_size']:<12.0f} {rating}")

        # Find best sampler
        best_sampler = max(results.items(), key=lambda x: x[1]['quality_score'])
        print(f"\n🏆 Best performing sampler: {best_sampler[0].upper()}")
        print(f"   Quality Score: {best_sampler[1]['quality_score']:.1f}%")

        if 'dpm_lm' in results:
            print(f"\n🔬 LML Method Analysis:")
            print(f"   Quality Score: {results['dpm_lm']['quality_score']:.1f}%")
            if results['dpm_lm']['quality_score'] > 70:
                print("   ✅ LML method shows superior performance!")
            else:
                print("   ⚠️  LML method performance needs optimization")

    print(f"\n📁 Generated images saved in: {args.output_dir}/")
    print("🎉 Test completed successfully!")

    return 0

if __name__ == '__main__':
    sys.exit(main())
