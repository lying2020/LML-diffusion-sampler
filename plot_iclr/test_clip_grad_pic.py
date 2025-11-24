"""
CLIP 模型梯度热力图可视化脚本
在 image_04621.jpg 上显示 CLIP ViT 模型的梯度热力图
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib
# 使用非交互式后端，避免 Qt 错误
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
import os
import sys

# 尝试导入 CLIP
try:
    import clip
except ImportError:
    print("Error: clip package not found. Please install it with: pip install clip-by-openai")
    sys.exit(1)

# 尝试从 HuggingFace 加载，如果失败则使用标准 CLIP
try:
    from transformers import CLIPModel, CLIPProcessor
    USE_HF = True
except ImportError:
    USE_HF = False
    print("Warning: transformers not found. Will use standard CLIP loading.")


def load_clip_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载 CLIP 模型
    优先尝试从 HuggingFace 格式路径加载，否则使用标准 CLIP 加载
    """
    processor = None
    if USE_HF and os.path.exists(model_path):
        try:
            print(f"尝试从 HuggingFace 格式加载模型: {model_path}")
            model = CLIPModel.from_pretrained(model_path)
            processor = CLIPProcessor.from_pretrained(model_path)
            model = model.to(device)
            model.eval()

            # 创建兼容的预处理函数
            def preprocess_fn(image):
                return processor(images=image, return_tensors="pt")['pixel_values'].to(device)

            return model, preprocess_fn, processor, True  # 返回 (model, preprocess, processor, is_hf)
        except Exception as e:
            print(f"HuggingFace 加载失败: {e}")
            print("尝试使用标准 CLIP 加载...")

    # 使用标准 CLIP 加载
    print("使用标准 CLIP 加载 ViT-L/14 模型...")
    model, preprocess = clip.load("ViT-L/14", device=device)

    # 确保模型参数为 float32
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

    return model, preprocess, processor, False  # 返回 (model, preprocess, processor, is_hf)


def load_image(image_path):
    """加载图片"""
    image = Image.open(image_path).convert('RGB')
    return image


def compute_gradcam(model, image_tensor, text_features, is_hf=False, device='cuda'):
    """
    计算 CLIP 模型的梯度热力图 (使用输入梯度方法)

    Args:
        model: CLIP 模型
        image_tensor: 预处理后的图片张量
        text_features: 文本特征（用于计算相似度）
        is_hf: 是否为 HuggingFace 格式模型
        device: 设备

    Returns:
        gradcam: 梯度热力图
    """
    model.eval()

    # 启用梯度计算
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
    image_tensor.requires_grad = True

    if is_hf:
        # HuggingFace 格式
        outputs = model.get_image_features(pixel_values=image_tensor)
        image_features = outputs / outputs.norm(dim=-1, keepdim=True)

        # 计算相似度
        # HuggingFace CLIP 可能没有 logit_scale，使用默认值 100.0
        if hasattr(model, 'logit_scale') and model.logit_scale is not None:
            logit_scale = model.logit_scale.exp()
        else:
            logit_scale = 100.0  # CLIP 默认 logit scale
        logits = (image_features @ text_features.t()) * logit_scale
    else:
        # 标准 CLIP 格式
        # 获取图像编码器
        image_encoder = model.visual

        # 前向传播获取特征
        image_features = image_encoder(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        logits = model.logit_scale.exp() * image_features @ text_features.t()

    # 选择最大相似度作为目标
    target = logits.max()

    # 反向传播
    model.zero_grad()
    target.backward(retain_graph=True)

    # 获取输入梯度
    gradients = image_tensor.grad.data

    # 计算梯度的绝对值并平均到单通道
    # gradients shape: (1, 3, H, W)
    gradcam = torch.abs(gradients).mean(dim=1).squeeze(0)  # (H, W)

    # 归一化到 [0, 1]
    gradcam_min = gradcam.min()
    gradcam_max = gradcam.max()
    if gradcam_max > gradcam_min:
        gradcam = (gradcam - gradcam_min) / (gradcam_max - gradcam_min + 1e-8)
    else:
        gradcam = torch.zeros_like(gradcam)

    return gradcam.cpu().numpy()


def get_text_features(model, text_prompts, processor=None, is_hf=False, device='cuda'):
    """
    获取文本特征

    Args:
        model: CLIP 模型
        text_prompts: 文本提示列表
        processor: HuggingFace processor (仅当 is_hf=True 时需要)
        is_hf: 是否为 HuggingFace 格式模型
        device: 设备

    Returns:
        text_features: 文本特征
    """
    if is_hf:
        # HuggingFace 格式
        if processor is None:
            raise ValueError("processor is required for HuggingFace format")
        inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    else:
        # 标准 CLIP 格式
        text_tokens = clip.tokenize(text_prompts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def resize_gradcam(gradcam, target_size):
    """
    将梯度热力图调整到目标尺寸

    Args:
        gradcam: 梯度热力图 (H, W)
        target_size: 目标尺寸 (height, width)

    Returns:
        resized_gradcam: 调整后的梯度热力图
    """
    gradcam_uint8 = (gradcam * 255).astype(np.uint8)
    resized = cv2.resize(gradcam_uint8, target_size, interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


def apply_colormap(gradcam, colormap_name='jet'):
    """
    将梯度热力图应用颜色映射

    Args:
        gradcam: 梯度热力图 (H, W)
        colormap_name: 颜色映射名称

    Returns:
        colored_gradcam: 彩色热力图 (H, W, 3)
    """
    # 使用新的 API 避免弃用警告
    try:
        # matplotlib >= 3.7
        cmap = matplotlib.colormaps[colormap_name]
    except (AttributeError, KeyError):
        # 兼容旧版本
        try:
            cmap = matplotlib.colormaps.get_cmap(colormap_name)
        except (AttributeError, KeyError):
            # 最后的回退
            cmap = cm.get_cmap(colormap_name)
    colored = cmap(gradcam)[:, :, :3]  # 移除 alpha 通道
    return (colored * 255).astype(np.uint8)


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    将热力图叠加到原图上

    Args:
        image: 原图 (PIL Image 或 numpy array)
        heatmap: 热力图 (numpy array, H, W, 3)
        alpha: 叠加透明度

    Returns:
        overlayed: 叠加后的图片
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 确保热力图尺寸与原图一致
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 叠加
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlayed


def visualize_gradcam(image_path, model_path, output_path=None, text_prompts=None, device='cuda'):
    """
    可视化 CLIP 模型的梯度热力图

    Args:
        image_path: 图片路径
        model_path: 模型路径
        output_path: 输出路径（可选）
        text_prompts: 文本提示列表（可选，默认使用通用提示）
        device: 设备
    """
    # 设置设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        device = 'cpu'

    # 加载模型
    print("加载 CLIP 模型...")
    model, preprocess, processor, is_hf = load_clip_model(model_path, device)

    # 加载图片
    print(f"加载图片: {image_path}")
    image = load_image(image_path)
    original_size = image.size  # (width, height)

    # 预处理图片
    if is_hf:
        image_tensor = preprocess(image)
        # HuggingFace 格式可能已经是 (1, 3, H, W)，确保需要梯度
        if image_tensor.requires_grad is False:
            image_tensor = image_tensor.clone().detach().requires_grad_(True)
    else:
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_tensor.requires_grad = True

    # 准备文本提示
    if text_prompts is None:
        # 默认使用通用提示
        text_prompts = [
            "a photo of a cat",
            "a photo of a dog",
            "a photo of a bird",
            "a photo of a car",
            "a photo of a person"
        ]

    print("计算文本特征...")
    text_features = get_text_features(model, text_prompts, processor, is_hf, device)

    # 计算梯度热力图
    print("计算梯度热力图...")
    gradcam = compute_gradcam(model, image_tensor, text_features, is_hf, device)

    # 调整梯度热力图尺寸到原图大小
    print("调整热力图尺寸...")
    gradcam_resized = resize_gradcam(gradcam, (original_size[1], original_size[0]))

    # 应用颜色映射
    print("应用颜色映射...")
    colored_heatmap = apply_colormap(gradcam_resized, 'jet')

    # 叠加到原图
    print("叠加热力图到原图...")
    overlayed = overlay_heatmap(image, colored_heatmap, alpha=0.5)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原图
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # 热力图
    axes[1].imshow(colored_heatmap)
    axes[1].set_title('Gradient Heatmap', fontsize=14)
    axes[1].axis('off')

    # 叠加图
    axes[2].imshow(overlayed)
    axes[2].set_title('Overlayed Image', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()

    # 保存结果
    if output_path is None:
        output_path = image_path.replace('.jpg', '_gradcam.jpg').replace('.png', '_gradcam.png')

    print(f"保存结果到: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存

    print("完成！")
    print(f"结果已保存到: {output_path}")


if __name__ == '__main__':
    # 配置路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_path, 'image_04621.jpg')
    model_path = '/home/liying/Documents/clip-vit-large-patch14'
    output_path = os.path.join(current_path, 'image_04621_gradcam.jpg')

    # 可选：自定义文本提示
    # text_prompts = ["a photo of a cat", "a photo of a dog"]

    # 运行可视化
    visualize_gradcam(
        image_path=image_path,
        model_path=model_path,
        output_path=output_path,
        text_prompts=None,  # 使用默认提示
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
