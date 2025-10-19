#!/bin/bash

# FID计算测试脚本
# 计算CIFAR-10真实图片与生成图片之间的FID值

# 设置路径
REAL_IMAGES_PATH="data/cifar10/cifar10_images"
OUTPUT_BASE_PATH="output/cifar10"
RESULTS_FILE="fid_results_table.txt"

# 定义步数
STEPS=(5 6 7 8 9 10 12 15 20 30 50 80)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."

    # 检查Python环境
    if ! command -v python &> /dev/null; then
        print_error "Python未找到，请确保Python已安装"
        exit 1
    fi

    # 检查pytorch-fid
    if ! python -c "import pytorch_fid" &> /dev/null; then
        print_warning "pytorch-fid未安装，正在安装..."
        pip install pytorch-fid
        if [ $? -ne 0 ]; then
            print_error "pytorch-fid安装失败"
            exit 1
        fi
    fi

    print_success "依赖检查完成"
}

# 检查路径是否存在
check_paths() {
    print_info "检查路径..."

    if [ ! -d "$REAL_IMAGES_PATH" ]; then
        print_error "真实图片路径不存在: $REAL_IMAGES_PATH"
        exit 1
    fi

    if [ ! -d "$OUTPUT_BASE_PATH" ]; then
        print_error "输出路径不存在: $OUTPUT_BASE_PATH"
        exit 1
    fi

    print_success "路径检查完成"
}

# 获取所有方法名称
get_methods() {
    local steps_dir="$1"
    if [ -d "$steps_dir" ]; then
        find "$steps_dir" -maxdepth 1 -type d -name "steps_*" | while read -r step_dir; do
            if [ -d "$step_dir" ]; then
                find "$step_dir" -maxdepth 1 -type d ! -path "$step_dir" | while read -r method_dir; do
                    basename "$method_dir"
                done
            fi
        done | sort -u
    fi
}

# 计算FID
calculate_fid() {
    local real_path="$1"
    local gen_path="$2"
    local method="$3"
    local step="$4"

    print_info "计算FID: $method (步数: $step)"

    # 检查生成图片路径是否存在
    if [ ! -d "$gen_path" ]; then
        print_warning "生成图片路径不存在: $gen_path"
        return 1
    fi

    # 检查生成图片数量
    local gen_count=$(find "$gen_path" -name "*.png" | wc -l)
    if [ "$gen_count" -eq 0 ]; then
        print_warning "生成图片路径中没有PNG文件: $gen_path"
        return 1
    fi

    print_info "找到 $gen_count 张生成图片"

    # 计算FID
    local fid_output=$(python -m pytorch_fid "$real_path" "$gen_path" 2>&1)
    local fid_value=$(echo "$fid_output" | grep "FID:" | awk '{print $2}')

    if [ -z "$fid_value" ]; then
        print_error "FID计算失败: $fid_output"
        return 1
    fi

    print_success "FID计算完成: $fid_value"
    echo "$fid_value"
}

# 创建结果表格
create_results_table() {
    local results_file="$1"

    print_info "创建结果表格: $results_file"

    # 创建表格头
    cat > "$results_file" << TABLE_HEADER_EOF
# CIFAR-10 FID Results Table
# Generated on: $(date)
# Real images path: $REAL_IMAGES_PATH
# Generated images path: $OUTPUT_BASE_PATH

TABLE_HEADER_EOF

    # 获取所有方法
    local all_methods=($(get_methods "$OUTPUT_BASE_PATH"))

    if [ ${#all_methods[@]} -eq 0 ]; then
        print_error "未找到任何方法"
        return 1
    fi

    print_info "找到方法: ${all_methods[*]}"

    # 创建表格
    printf "%-15s" "Method" >> "$results_file"
    for step in "${STEPS[@]}"; do
        printf "  %-8s" "Step $step" >> "$results_file"
    done
    printf "\n" >> "$results_file"

    # 添加分隔线
    printf "%-15s" "---------------" >> "$results_file"
    for step in "${STEPS[@]}"; do
        printf "  %-8s" "--------" >> "$results_file"
    done
    printf "\n" >> "$results_file"

    # 为每个方法计算FID
    for method in "${all_methods[@]}"; do
        printf "%-15s" "$method" >> "$results_file"

        for step in "${STEPS[@]}"; do
            local gen_path="$OUTPUT_BASE_PATH/steps_$step/$method"
            local fid_value=$(calculate_fid "$REAL_IMAGES_PATH" "$gen_path" "$method" "$step")

            if [ $? -eq 0 ] && [ -n "$fid_value" ]; then
                printf "  %-8.2f" "$fid_value" >> "$results_file"
            else
                printf "  %-8s" "N/A" >> "$results_file"
            fi
        done

        printf "\n" >> "$results_file"
    done

    print_success "结果表格已保存到: $results_file"
}

# 生成CSV格式的结果
create_csv_results() {
    local results_file="$1"
    local csv_file="${results_file%.txt}.csv"

    print_info "生成CSV格式结果: $csv_file"

    # 转换表格为CSV格式
    awk '
    BEGIN { FS = "  +"; OFS = "," }
    {
        gsub(/^[ \t]+|[ \t]+$/, "", $0)  # 去除首尾空格
        gsub(/[ \t]+/, ",", $0)          # 将多个空格替换为逗号
        print $0
    }' "$results_file" > "$csv_file"

    print_success "CSV结果已保存到: $csv_file"
}

# 生成统计报告
generate_summary() {
    local results_file="$1"
    local summary_file="${results_file%.txt}_summary.txt"

    print_info "生成统计报告: $summary_file"

    cat > "$summary_file" << SUMMARY_HEADER_EOF
# CIFAR-10 FID Results Summary
# Generated on: $(date)

## 实验配置
- 真实图片路径: $REAL_IMAGES_PATH
- 生成图片路径: $OUTPUT_BASE_PATH
- 测试步数: ${STEPS[*]}
- 结果文件: $results_file

## 统计信息
SUMMARY_HEADER_EOF

    # 添加统计信息
    python3 << PYTHON_CODE_EOF
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
PYTHON_CODE_EOF "$results_file" >> "$summary_file"

    print_success "统计报告已保存到: $summary_file"
}

# 主函数
main() {
    print_info "开始CIFAR-10 FID计算测试"
    print_info "================================"

    # 检查依赖
    check_dependencies

    # 检查路径
    check_paths

    # 创建结果表格
    create_results_table "$RESULTS_FILE"

    # 生成CSV格式
    create_csv_results "$RESULTS_FILE"

    # 生成统计报告
    generate_summary "$RESULTS_FILE"

    print_success "FID计算测试完成！"
    print_info "结果文件:"
    print_info "  - 表格格式: $RESULTS_FILE"
    print_info "  - CSV格式: ${RESULTS_FILE%.txt}.csv"
    print_info "  - 统计报告: ${RESULTS_FILE%.txt}_summary.txt"
}

# 运行主函数
main "$@"
