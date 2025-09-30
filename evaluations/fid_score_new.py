"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
import hashlib
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from pytorch_fid.inception import InceptionV3

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

# 全局缓存目录
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".fid_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 配置参数
MAX_IMAGES = 10000  # 最大图片数量限制

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1, max_images=10000):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    -- max_images  : Maximum number of images to process

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # Limit to maximum images for efficiency
    if len(files) > max_images:
        print(f"Found {len(files)} images, limiting to {max_images} for efficiency")
        # Use random sampling to select a representative subset
        import random
        random.seed(42)  # For reproducibility
        files = random.sample(files, max_images)
        files.sort()  # Keep sorted order

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    coco_transform = TF.Compose([
     TF.Resize(512),
     TF.CenterCrop(512),
     TF.ToTensor()])

    dataset = ImagePathDataset(files, transforms=coco_transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1, max_images=10000):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    -- max_images  : Maximum number of images to process

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers, max_images)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_path_hash(path, dims=2048):
    """Generate a hash for a path based on its contents and parameters."""
    path = pathlib.Path(path)

    # Get all image files
    files = sorted([file for ext in IMAGE_EXTENSIONS
                   for file in path.glob('**/*.{}'.format(ext))])

    if not files:
        return None

    # Create hash based on file paths, sizes, and parameters
    hash_input = f"{dims}_{len(files)}_"
    for file_path in files[:100]:  # Use first 100 files for hash
        try:
            stat = file_path.stat()
            hash_input += f"{file_path}_{stat.st_size}_{stat.st_mtime}_"
        except:
            continue

    return hashlib.md5(hash_input.encode()).hexdigest()


def get_cache_path(path, dims=2048):
    """Get the cache file path for a given directory."""
    path_hash = get_path_hash(path, dims)
    if path_hash is None:
        return None
    return os.path.join(CACHE_DIR, f"fid_stats_{path_hash}_{dims}.npz")


def load_cached_stats(cache_path):
    """Load cached statistics if available."""
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path) as f:
                return f['mu'][:], f['sigma'][:]
        except:
            return None
    return None


def save_cached_stats(cache_path, mu, sigma):
    """Save statistics to cache."""
    try:
        np.savez_compressed(cache_path, mu=mu, sigma=sigma)
        return True
    except:
        return False


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1, use_cache=True, max_images=10000):
    """Compute statistics with caching support."""
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
        return m, s

    # Check cache first
    if use_cache:
        cache_path = get_cache_path(path, dims)
        if cache_path:
            cached_stats = load_cached_stats(cache_path)
            if cached_stats is not None:
                print(f"Using cached statistics for {path}")
                return cached_stats

    # Compute statistics
    path = pathlib.Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                   for file in path.glob('**/*.{}'.format(ext))])

    if not files:
        raise ValueError(f"No image files found in {path}")

    # Limit to maximum images for efficiency
    if len(files) > max_images:
        print(f"Found {len(files)} images, limiting to {max_images} for efficiency")
        # Use random sampling to select a representative subset
        import random
        random.seed(42)  # For reproducibility
        files = random.sample(files, max_images)
        files.sort()  # Keep sorted order

    print(f"Computing statistics for {len(files)} images in {path}")
    m, s = calculate_activation_statistics(files, model, batch_size,
                                           dims, device, num_workers, max_images)

    # Save to cache
    if use_cache and cache_path:
        if save_cached_stats(cache_path, m, s):
            print(f"Statistics cached to {cache_path}")

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1,
                             use_cache=True, output='', max_images=10000):
    """Calculates the FID of two paths with caching support."""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    print(f"Computing statistics for path 1: {paths[0]}")
    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers, use_cache, max_images)

    print(f"Computing statistics for path 2: {paths[1]}")
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers, use_cache, max_images)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def save_fid_stats(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    if not os.path.exists(paths[0]):
        raise RuntimeError('Invalid path: %s' % paths[0])

    if os.path.exists(paths[1]):
        raise RuntimeError('Existing output file: %s' % paths[1])

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    print(f"Saving statistics for {paths[0]}")
    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers, use_cache=False)
    np.savez_compressed(paths[1], mu=m1, sigma=s1)


def batch_calculate_fid(real_images_path, output_base_path, steps,
                       batch_size=50, device='cuda', dims=2048,
                       num_workers=1, use_cache=True, max_images=10000):
    """批量计算FID值"""
    results = {}

    # 确保真实图片路径存在
    if not os.path.exists(real_images_path):
        raise ValueError(f"Real images path does not exist: {real_images_path}")

    print(f"Real images path: {real_images_path}")
    print(f"Output base path: {output_base_path}")
    print(f"Steps to evaluate: {steps}")
    print(f"Using cache: {use_cache}")
    print(f"Max images per method: {max_images}")
    print("=" * 60)

    # 获取所有方法
    all_methods = set()
    for step in steps:
        step_dir = os.path.join(output_base_path, f"steps_{step}")
        if os.path.exists(step_dir):
            for item in os.listdir(step_dir):
                item_path = os.path.join(step_dir, item)
                if os.path.isdir(item_path):
                    all_methods.add(item)

    all_methods = sorted(list(all_methods))
    print(f"Found methods: {all_methods}")

    # 为每个方法计算FID
    for method in all_methods:
        results[method] = {}
        print(f"\nProcessing method: {method}")

        for step in steps:
            gen_path = os.path.join(output_base_path, f"steps_{step}", method)

            if not os.path.exists(gen_path):
                print(f"  Step {step}: Path not found, skipping")
                results[method][step] = None
                continue

            try:
                print(f"  Step {step}: Computing FID...")
                fid_value = calculate_fid_given_paths(
                    [real_images_path, gen_path],
                    batch_size, device, dims, num_workers, use_cache, max_images
                )
                results[method][step] = float(fid_value)
                print(f"  Step {step}: FID = {fid_value:.2f}")
            except Exception as e:
                print(f"  Step {step}: Error - {e}")
                results[method][step] = None

    return results


def save_results_table(results, steps, output_file):
    """保存结果表格"""
    with open(output_file, 'w') as f:
        f.write("# CIFAR-10 FID Results Table\n")
        f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Lower FID values indicate better quality\n\n")

        # 表头
        f.write(f"{'Method':<15}")
        for step in steps:
            f.write(f"  {f'Step {step}':<8}")
        f.write("\n")

        # 分隔线
        f.write("-" * (15 + 11 * len(steps)))
        f.write("\n")

        # 数据行
        for method, step_results in results.items():
            f.write(f"{method:<15}")
            for step in steps:
                if step in step_results and step_results[step] is not None:
                    f.write(f"  {step_results[step]:<8.2f}")
                else:
                    f.write(f"  {'N/A':<8}")
            f.write("\n")


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--save-stats', action='store_true',
                        help=('Generate an npz archive from a directory of samples. '
                              'The first path is used as input and the second as output.'))
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='Use cached statistics to avoid recomputation')
    parser.add_argument('--batch-mode', action='store_true',
                        help='Run in batch mode for multiple methods and steps')
    parser.add_argument('--real-path', type=str, default='data/cifar10/cifar10_images',
                        help='Path to real images (for batch mode)')
    parser.add_argument('--output-path', type=str, default='output/cifar10',
                        help='Path to generated images (for batch mode)')
    parser.add_argument('--steps', type=int, nargs='+',
                        default=[5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 80],
                        help='Steps to evaluate (for batch mode)')
    parser.add_argument('--output-file', type=str, default='fid_results.txt',
                        help='Output file for results (for batch mode)')
    parser.add_argument('--max-images', type=int, default=10000,
                        help='Maximum number of images to process (default: 10000)')
    parser.add_argument('path', type=str, nargs='*',
                        help=('Paths to the generated images or '
                              'to .npz statistic files'))

    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    if args.save_stats:
        if len(args.path) != 2:
            raise ValueError("Need exactly 2 paths for save-stats mode")
        save_fid_stats(args.path, args.batch_size, device, args.dims, num_workers)
        return

    if args.batch_mode:
        # 批量模式
        results = batch_calculate_fid(
            args.real_path, args.output_path, args.steps,
            args.batch_size, device, args.dims, num_workers, args.use_cache, args.max_images
        )
        save_results_table(results, args.steps, args.output_file)
        print(f"\nResults saved to: {args.output_file}")
    else:
        # 单次计算模式
        if len(args.path) != 2:
            raise ValueError("Need exactly 2 paths for FID calculation")

        fid_value = calculate_fid_given_paths(
            args.path, args.batch_size, device, args.dims,
            num_workers, args.use_cache, args.max_images
        )
        print('FID: ', fid_value)


if __name__ == '__main__':
    main()
