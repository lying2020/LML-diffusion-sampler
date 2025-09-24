#!/usr/bin/env python3
"""
FID计算示例脚本
演示如何使用转换后的CIFAR-10图片进行FID计算
"""

import os
import sys
import subprocess
import argparse

def run_fid_calculation(real_path, generated_path, batch_size=32, device='cuda'):
    """
    运行FID计算
    
    Args:
        real_path: 真实图片路径
        generated_path: 生成图片路径
        batch_size: 批次大小
        device: 计算设备
    """
    
    # 检查路径是否存在
    if not os.path.exists(real_path):
        print(f"错误: 真实图片路径 {real_path} 不存在!")
        return None
        
    if not os.path.exists(generated_path):
        print(f"错误: 生成图片路径 {generated_path} 不存在!")
        return None
    
    # 构建命令
    cmd = [
        'python3', '../evaluations/fid_score.py',
        real_path, generated_path,
        '--batch-size', str(batch_size),
        '--device', device
    ]
    
    print(f"运行FID计算...")
    print(f"真实图片: {real_path}")
    print(f"生成图片: {generated_path}")
    print(f"批次大小: {batch_size}")
    print(f"设备: {device}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            # 提取FID值
            output_lines = result.stdout.strip().split('\n')
            fid_line = [line for line in output_lines if line.startswith('FID:')]
            if fid_line:
                fid_value = float(fid_line[0].split(':')[1].strip())
                print(f"FID计算成功!")
                print(f"FID值: {fid_value}")
                return fid_value
            else:
                print("无法解析FID值")
                return None
        else:
            print(f"FID计算失败:")
            print(f"错误输出: {result.stderr}")
            return None
    except Exception as e:
        print(f"运行FID计算时出错: {e}")
        return None

def save_fid_stats(image_path, output_file, batch_size=32, device='cuda'):
    """
    保存FID统计信息
    
    Args:
        image_path: 图片路径
        output_file: 输出统计文件路径
        batch_size: 批次大小
        device: 计算设备
    """
    
    if not os.path.exists(image_path):
        print(f"错误: 图片路径 {image_path} 不存在!")
        return False
    
    cmd = [
        'python3', '../evaluations/fid_score.py',
        '--save-stats',
        image_path, output_file,
        '--batch-size', str(batch_size),
        '--device', device
    ]
    
    print(f"保存FID统计信息...")
    print(f"图片路径: {image_path}")
    print(f"输出文件: {output_file}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            print("统计信息保存成功!")
            return True
        else:
            print(f"保存统计信息失败:")
            print(f"错误输出: {result.stderr}")
            return False
    except Exception as e:
        print(f"保存统计信息时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='FID计算示例')
    parser.add_argument('--real_path', type=str, default='cifar10_test_images/test',
                        help='真实图片路径')
    parser.add_argument('--generated_path', type=str, required=True,
                        help='生成图片路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--save_stats', action='store_true',
                        help='保存真实图片的统计信息')
    
    args = parser.parse_args()
    
    print("=== FID计算示例 ===")
    
    # 保存统计信息（如果需要）
    if args.save_stats:
        stats_file = f"{args.real_path}_stats.npz"
        if save_fid_stats(args.real_path, stats_file, args.batch_size, args.device):
            print(f"统计信息已保存到: {stats_file}")
        else:
            print("保存统计信息失败，继续使用直接计算...")
    
    # 计算FID
    fid_value = run_fid_calculation(
        args.real_path, 
        args.generated_path, 
        args.batch_size, 
        args.device
    )
    
    if fid_value is not None:
        print(f"\n=== 结果 ===")
        print(f"FID值: {fid_value}")
        
        # 提供一些解释
        if fid_value < 10:
            print("FID值 < 10: 生成图片质量很高，与真实图片非常相似")
        elif fid_value < 50:
            print("FID值 < 50: 生成图片质量良好")
        elif fid_value < 100:
            print("FID值 < 100: 生成图片质量一般")
        else:
            print("FID值 >= 100: 生成图片质量较差，需要改进模型")
    else:
        print("FID计算失败!")

if __name__ == '__main__':
    main()
