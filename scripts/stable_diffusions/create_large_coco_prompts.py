#!/usr/bin/env python3
"""
Create a larger COCO-style prompts dataset for testing
"""
import sys
import json
import random
import os
sys.path.append(os.getcwd())

import project as project


def create_large_coco_prompts(num_prompts=1000):
    """Create a larger dataset of COCO-style prompts"""

    # 基础场景和物体
    scenes = [
        "a beautiful landscape", "a busy city street", "a peaceful garden",
        "a modern kitchen", "a cozy living room", "a sunny beach",
        "a mountain peak", "a forest path", "a lake shore", "a park bench"
    ]

    objects = [
        "cat", "dog", "car", "bicycle", "tree", "flower", "book", "phone",
        "chair", "table", "window", "door", "building", "bridge", "boat",
        "airplane", "train", "bus", "motorcycle", "horse", "bird", "fish"
    ]

    actions = [
        "sitting", "standing", "walking", "running", "playing", "eating",
        "reading", "sleeping", "working", "cooking", "driving", "flying"
    ]

    colors = [
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
        "black", "white", "gray", "silver", "golden"
    ]

    prompts_dict = {}

    for i in range(num_prompts):
        key = f"{i:05d}"

        # 随机组合生成提示词
        scene = random.choice(scenes)
        obj = random.choice(objects)
        action = random.choice(actions)
        color = random.choice(colors)

        # 生成不同类型的提示词
        prompt_types = [
            f"{scene} with a {color} {obj}",
            f"a {color} {obj} {action} in {scene}",
            f"{scene} where a {obj} is {action}",
            f"a {obj} {action} near {scene}",
            f"{scene} featuring a {color} {obj} that is {action}"
        ]

        prompt = random.choice(prompt_types)
        prompts_dict[key] = prompt

    return prompts_dict

def main():
    print("创建大型 COCO 风格提示词数据集...")

    # 创建1000个提示词
    prompts_dict = create_large_coco_prompts(1000)

    # 保存到文件
    output_file = os.path.join(project.project_dir, "evaluations", "coco_prompts", "fid_3W_json.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts_dict, f, indent=2, ensure_ascii=False)

    print(f"已创建 {len(prompts_dict)} 个提示词")
    print(f"保存到: {output_file}")

    # 显示前10个示例
    print("\n前10个提示词示例:")
    for i, (key, prompt) in enumerate(list(prompts_dict.items())[:10]):
        print(f"{key}: {prompt}")

if __name__ == "__main__":
    main()
