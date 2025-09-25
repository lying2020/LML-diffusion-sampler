
import sys
import re

def parse_fid_file(filename):
    """解析FID结果文件"""
    methods = []
    steps = []
    fid_values = {}

    with open(filename, 'r') as f:
        lines = f.readlines()

    # 找到表头行
    header_line = None
    for i, line in enumerate(lines):
        if 'Step' in line and 'Method' in line:
            header_line = i
            break

    if header_line is None:
        print("未找到表头")
        return

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

def generate_summary(methods, steps, fid_values):
    """生成统计报告"""
    print(f"\n## 方法统计")
    print(f"- 总方法数: {len(methods)}")
    print(f"- 方法列表: {', '.join(methods)}")

    print(f"\n## 步数统计")
    print(f"- 总步数: {len(steps)}")
    print(f"- 步数列表: {', '.join(map(str, steps))}")

    print(f"\n## FID值统计")
    for method in methods:
        values = [v for v in fid_values[method] if v is not None]
        if values:
            print(f"\n### {method}")
            print(f"- 有效FID值数量: {len(values)}")
            print(f"- 最小FID值: {min(values):.2f}")
            print(f"- 最大FID值: {max(values):.2f}")
            print(f"- 平均FID值: {sum(values)/len(values):.2f}")
        else:
            print(f"\n### {method}")
            print(f"- 无有效FID值")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python script.py <results_file>")
        sys.exit(1)

    results_file = sys.argv[1]
    try:
        methods, steps, fid_values = parse_fid_file(results_file)
        generate_summary(methods, steps, fid_values)
    except Exception as e:
        print(f"错误: {e}")
