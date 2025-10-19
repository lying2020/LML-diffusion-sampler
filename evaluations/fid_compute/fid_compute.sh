#!/bin/bash

# FID计算测试脚本
# 计算CIFAR-10真实图片与生成图片之间的FID值

# 检查参数
if [ $# -ne 2 ]; then
    echo "用法: $0 <cifar10_real_images_path> <generated_images_path>"
    echo "示例: $0 data/cifar10/cifar10_images output/cifar10"
    exit 1
fi

# 设置路径
REAL_IMAGES_PATH="$1"
OUTPUT_BASE_PATH="$2"
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
    if ! command -v python3 &> /dev/null; then
        print_error "Python3未找到，请确保Python3已安装"
        exit 1
    fi

    # 检查fid_score.py文件
    if [ ! -f "evaluations/fid_score.py" ]; then
        print_error "evaluations/fid_score.py 文件不存在"
        exit 1
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

    # 使用evaluations/fid_score.py计算FID
    local fid_output=$(python3 evaluations/fid_score.py "$real_path" "$gen_path")
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
    cat > "$results_file" << TABLE_EOF
# CIFAR-10 FID Results Table
# Generated on: $(date)
# Real images path: $REAL_IMAGES_PATH
# Generated images path: $OUTPUT_BASE_PATH

TABLE_EOF

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

    if [ -f "fid_summary.py" ]; then
        print_info "生成统计报告..."
        python3 fid_summary.py "$results_file"
        if [ $? -eq 0 ]; then
            print_success "统计报告生成完成"
        else
            print_warning "统计报告生成失败"
        fi
    else
        print_warning "fid_summary.py 不存在，跳过统计报告生成"
    fi
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
