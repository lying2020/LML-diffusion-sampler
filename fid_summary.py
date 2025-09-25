#!/usr/bin/env python3
"""
FID结果统计报告生成器
解析FID结果文件并生成详细的统计报告
"""

import sys
import re
import os
from datetime import datetime

def parse_fid_file(filename):
    """解析FID结果文件"""
    methods = []
    steps = []
    fid_values = {}
    
    if not os.path.exists(filename):
        print(f"错误: 文件不存在 {filename}")
        return None, None, None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 找到表头行
    header_line = None
    for i, line in enumerate(lines):
        if 'Step' in line and 'Method' in line:
            header_line = i
            break
    
    if header_line is None:
        print("错误: 未找到表头")
        return None, None, None
    
    # 解析表头获取步数
    header = lines[header_line].strip()
    step_matches = re.findall(r'Step (\d+)', header)
    steps = [int(s) for s in step_matches]
    
    # 解析数据行
    for line in lines[header_line + 2:]:  # 跳过分隔线
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split()
            if len(parts) > 1:
                method = parts[0]
                methods.append(method)
                fid_values[method] = []
                
                for i, part in enumerate(parts[1:], 1):
                    if part != 'N/A' and part.replace('.', '').isdigit():
                        fid_values[method].append(float(part))
                    else:
                        fid_values[method].append(None)
    
    return methods, steps, fid_values

def generate_summary(methods, steps, fid_values, output_file):
    """生成统计报告"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# CIFAR-10 FID Results Summary\n")
        f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验配置\n")
        f.write(f"- 总方法数: {len(methods)}\n")
        f.write(f"- 方法列表: {', '.join(methods)}\n")
        f.write(f"- 总步数: {len(steps)}\n")
        f.write(f"- 步数列表: {', '.join(map(str, steps))}\n\n")
        
        f.write("## FID值统计\n")
        for method in methods:
            values = [v for v in fid_values[method] if v is not None]
            f.write(f"\n### {method}\n")
            if values:
                f.write(f"- 有效FID值数量: {len(values)}\n")
                f.write(f"- 最小FID值: {min(values):.2f}\n")
                f.write(f"- 最大FID值: {max(values):.2f}\n")
                f.write(f"- 平均FID值: {sum(values)/len(values):.2f}\n")
                
                # 找出最佳步数
                best_step_idx = values.index(min(values))
                best_step = steps[best_step_idx]
                f.write(f"- 最佳步数: {best_step} (FID: {min(values):.2f})\n")
            else:
                f.write("- 无有效FID值\n")
        
        # 总体统计
        f.write(f"\n## 总体统计\n")
        all_values = []
        for method_values in fid_values.values():
            all_values.extend([v for v in method_values if v is not None])
        
        if all_values:
            f.write(f"- 所有FID值总数: {len(all_values)}\n")
            f.write(f"- 全局最小FID值: {min(all_values):.2f}\n")
            f.write(f"- 全局最大FID值: {max(all_values):.2f}\n")
            f.write(f"- 全局平均FID值: {sum(all_values)/len(all_values):.2f}\n")
        else:
            f.write("- 无有效FID值\n")

def main():
    if len(sys.argv) != 2:
        print("用法: python3 fid_summary.py <results_file>")
        print("示例: python3 fid_summary.py fid_results_table.txt")
        sys.exit(1)
    
    results_file = sys.argv[1]
    summary_file = results_file.replace('.txt', '_summary.txt')
    
    try:
        methods, steps, fid_values = parse_fid_file(results_file)
        if methods is None:
            sys.exit(1)
        
        generate_summary(methods, steps, fid_values, summary_file)
        print(f"统计报告已生成: {summary_file}")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
