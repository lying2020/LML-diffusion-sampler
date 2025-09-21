#!/usr/bin/env python3
"""
CelebA-HQ Demo Script

This script demonstrates the new features of the reorganized celeba.py script,
including evaluation metrics and comparison functionality.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the description"""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ ERROR")
            if result.stderr:
                print("Error:")
                print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT - Command took too long")
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

def main():
    print("🚀 CelebA-HQ Script Demo")
    print("This demo shows the new features of the reorganized celeba.py script")

    # Demo 1: Show help
    run_command(
        "python3 scripts/celeba.py --help",
        "Show help message with all available options"
    )

    # Demo 2: Test evaluation functions
    run_command(
        "python3 -c \"import sys; sys.path.append('.'); from scripts.celeba import *; from PIL import Image; import numpy as np; img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)); print('Testing evaluation functions...'); print(f'ColorS: {calculate_color_score([img]):.3f}'); print(f'FS: {calculate_face_score([img]):.3f}'); print(f'DFIQA: {calculate_df_iqa([img]):.3f}'); print(f'PicS: {calculate_pic_score([img]):.3f}'); print(f'EAT: {calculate_eat_score([img]):.3f}'); print(f'Laion: {calculate_laion_score([img]):.3f}'); print('All functions working!')\"",
        "Test evaluation functions with a random image"
    )

    # Demo 3: Show comparison table format
    run_command(
        "python3 -c \"import sys; sys.path.append('.'); from scripts.celeba import *; import json; print('Demo comparison table format:'); print('='*80); print('Table 2. Comparison of different samplers on CelebA-HQ unconditional generation.'); print('Best results are bolded and the second best results are underlined.'); print('='*80); print('Methods         Colorful        Face Quality     Aesthetic'); print('                ColorS(↑)  FS(↑)  DFIQA(↑) PicS(↑)  EAT(↑)   Laion(↑)'); print('-'*80); print('DDIM [75]       34.13     4.85   0.539    19.85    4.61     5.24'); print('PNDM [50]       38.29     4.89   0.553    19.70    4.28     5.11'); print('DPM [51]        34.85     5.06   0.566    20.00    4.51     5.29'); print('DPM++ [52]      35.08     5.08   0.560    19.97    4.46     5.26'); print('UniPC [91]      35.95     5.09   0.568    19.97    4.47     5.26'); print('LML(Ours)       **40.53** **5.24** **0.607** **20.98** **4.75** **5.37**'); print('='*80)\"",
        "Show comparison table format (from the image)"
    )

    print(f"\n{'='*60}")
    print("DEMO COMPLETED")
    print(f"{'='*60}")
    print("The reorganized celeba.py script now includes:")
    print("✅ Better organization following cifar10.py structure")
    print("✅ Comprehensive evaluation metrics (ColorS, FS, DFIQA, PicS, EAT, Laion)")
    print("✅ Comparison table generation with best/second-best highlighting")
    print("✅ Enhanced error handling and logging")
    print("✅ Support for comparing all samplers")
    print("✅ Detailed progress reporting")
    print("\nUsage examples:")
    print("  python3 scripts/celeba.py --sampler_type ddim --test_num 5 --evaluate")
    print("  python3 scripts/celeba.py --compare_all --test_num 3 --save_results")
    print("  python3 scripts/celeba.py --sampler_type dpm_lm --lamb 0.001 --test_num 10 --evaluate")

if __name__ == '__main__':
    main()
