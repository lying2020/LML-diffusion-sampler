#!/usr/bin/env python3
"""
Optimized Hessian Testing for LML Diffusion Sampler

This script provides optimized versions of Hessian computation methods:
1. Original LML method (fastest)
2. Simplified Hessian-Free method (balanced)
3. Lightweight Hessian approximation (faster than explicit)
"""

import sys
import os
import time
import torch
import argparse
import numpy as np
from typing import Dict, List, Tuple
import json

sys.path.append(os.getcwd())
from diffusers import DDPMPipeline
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler

def lm_correct_lightweight(prev_noise, noise_pred, lamb, kappa, model, x, t):
    """
    Lightweight Hessian approximation using only diagonal elements
    Much faster than full Hessian computation
    """
    if prev_noise is not None:
        noise_pred_ema = kappa * prev_noise + (1 - kappa) * noise_pred
    else:
        noise_pred_ema = noise_pred
    
    # Compute gradient for diagonal Hessian approximation
    x_grad = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        score_pred = model(x_grad, t)
        if hasattr(score_pred, 'sample'):
            score_pred = score_pred.sample
        log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
        log_prob = log_prob.sum()
    
    # Compute gradient
    grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]
    
    # Simplified diagonal Hessian approximation
    # Use gradient magnitude as proxy for diagonal elements
    grad_mag = torch.abs(grad) + 1e-8
    hessian_diag_approx = grad_mag * 0.1  # Scaling factor
    
    # Apply correction
    correction_factor = 1.0 / (1.0 + lamb * hessian_diag_approx)
    corrected_noise = noise_pred * correction_factor
    
    # Normalize
    norm = torch.sqrt((noise_pred * noise_pred).sum(dim=(1, 2, 3), keepdim=True))
    norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
    corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)
    
    return corrected_noise

def lm_correct_hessian_free_fast(prev_noise, noise_pred, lamb, kappa, model, x, t):
    """
    Fast Hessian-Free method with simplified CG
    """
    if prev_noise is not None:
        noise_pred_ema = kappa * prev_noise + (1 - kappa) * noise_pred
    else:
        noise_pred_ema = noise_pred
    
    def hessian_vector_product_fast(v):
        """Simplified HVP computation"""
        v_grad = v.clone().detach().requires_grad_(True)
        x_grad = x.clone().detach().requires_grad_(True)
        
        with torch.enable_grad():
            score_pred = model(x_grad, t)
            if hasattr(score_pred, 'sample'):
                score_pred = score_pred.sample
            log_prob = -0.5 * torch.sum(score_pred ** 2, dim=(1, 2, 3))
            log_prob = log_prob.sum()
        
        grad = torch.autograd.grad(log_prob, x_grad, create_graph=True)[0]
        grad_dot_v = torch.sum(grad * v_grad)
        hv = torch.autograd.grad(grad_dot_v, x_grad, retain_graph=True)[0]
        return hv
    
    def conjugate_gradient_fast(b, max_iter=5, tol=1e-3):
        """Fast CG with fewer iterations"""
        x_cg = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        
        r_norm_sq = torch.sum(r ** 2)
        r_norm_0 = torch.sqrt(r_norm_sq)
        
        for i in range(max_iter):
            Hp = hessian_vector_product_fast(p)
            p_Hp = torch.sum(p * Hp)
            
            if p_Hp <= 0:
                break
            
            alpha = r_norm_sq / p_Hp
            x_cg = x_cg + alpha * p
            r = r - alpha * Hp
            
            r_norm_sq_new = torch.sum(r ** 2)
            r_norm = torch.sqrt(r_norm_sq_new)
            
            if r_norm < tol * r_norm_0:
                break
            
            beta = r_norm_sq_new / r_norm_sq
            p = r + beta * p
            r_norm_sq = r_norm_sq_new
        
        return x_cg
    
    # Solve H^{-1} * noise_pred using fast CG
    corrected_noise = conjugate_gradient_fast(noise_pred)
    
    # Add regularization
    corrected_noise = corrected_noise / (1.0 + lamb)
    
    # Normalize
    norm = torch.sqrt((noise_pred * noise_pred).sum(dim=(1, 2, 3), keepdim=True))
    norm_corrected = torch.sqrt((corrected_noise * corrected_noise).sum(dim=(1, 2, 3), keepdim=True))
    corrected_noise = corrected_noise * norm / (norm_corrected + 1e-8)
    
    return corrected_noise

class OptimizedHessianTester:
    """Optimized testing suite for Hessian methods"""
    
    def __init__(self, model_id: str, device: str = 'cuda'):
        self.model_id = model_id
        self.device = device
        self.results = {}
        
        # Load model
        print("Loading model...")
        self.pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        self.pipe.unet.to(device)
        
    def test_original_lml(self, test_num: int = 10) -> Dict:
        """Test original LML method"""
        print(f"\n{'='*60}")
        print("Testing: Original LML Method")
        print(f"{'='*60}")
        
        # Configure scheduler
        self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.solver_order = 3
        self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        self.pipe.scheduler.lamb = 0.0008
        self.pipe.scheduler.lm = True
        self.pipe.scheduler.kappa = 1.0e-8
        
        return self._generate_and_measure(test_num, "original")
    
    def test_lightweight_hessian(self, test_num: int = 10) -> Dict:
        """Test lightweight Hessian approximation"""
        print(f"\n{'='*60}")
        print("Testing: Lightweight Hessian Approximation")
        print(f"{'='*60}")
        
        # Configure scheduler
        self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.solver_order = 3
        self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        self.pipe.scheduler.lamb = 0.0008
        self.pipe.scheduler.lm = True
        self.pipe.scheduler.kappa = 1.0e-8
        
        # Override the lm_correct function
        original_lm_correct = self.pipe.scheduler.lm_correct
        
        def lightweight_lm_correct(prev_noise, noise_pred, lamb, kappa):
            return lm_correct_lightweight(prev_noise, noise_pred, lamb, kappa, 
                                        self.pipe.unet, self.pipe.scheduler.prev_sample, 
                                        self.pipe.scheduler.timestep)
        
        self.pipe.scheduler.lm_correct = lightweight_lm_correct
        
        result = self._generate_and_measure(test_num, "lightweight")
        
        # Restore original function
        self.pipe.scheduler.lm_correct = original_lm_correct
        
        return result
    
    def test_fast_hessian_free(self, test_num: int = 10) -> Dict:
        """Test fast Hessian-Free method"""
        print(f"\n{'='*60}")
        print("Testing: Fast Hessian-Free Method")
        print(f"{'='*60}")
        
        # Configure scheduler
        self.pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.config.solver_order = 3
        self.pipe.scheduler.config.algorithm_type = "dpmsolver"
        self.pipe.scheduler.lamb = 0.0008
        self.pipe.scheduler.lm = True
        self.pipe.scheduler.kappa = 1.0e-8
        
        # Override the lm_correct function
        original_lm_correct = self.pipe.scheduler.lm_correct
        
        def fast_hessian_free_lm_correct(prev_noise, noise_pred, lamb, kappa):
            return lm_correct_hessian_free_fast(prev_noise, noise_pred, lamb, kappa, 
                                              self.pipe.unet, self.pipe.scheduler.prev_sample, 
                                              self.pipe.scheduler.timestep)
        
        self.pipe.scheduler.lm_correct = fast_hessian_free_lm_correct
        
        result = self._generate_and_measure(test_num, "hessian_free_fast")
        
        # Restore original function
        self.pipe.scheduler.lm_correct = original_lm_correct
        
        return result
    
    def _generate_and_measure(self, test_num: int, method_name: str) -> Dict:
        """Generate images and measure performance"""
        generation_times = []
        image_qualities = []
        
        print(f"Generating {test_num} images...")
        start_time = time.time()
        
        with torch.no_grad():
            for seed in range(test_num):
                torch.manual_seed(seed)
                
                step_start = time.time()
                images = self.pipe(batch_size=1, num_inference_steps=20).images
                step_time = time.time() - step_start
                generation_times.append(step_time)
                
                # Simple quality metric
                if images:
                    img_array = np.array(images[0])
                    quality = np.var(img_array)
                    image_qualities.append(quality)
                
                if seed % 5 == 0:
                    print(f"  Generated {seed+1}/{test_num} images")
        
        total_time = time.time() - start_time
        
        result = {
            'method': method_name,
            'total_time': total_time,
            'avg_time_per_image': np.mean(generation_times),
            'std_time_per_image': np.std(generation_times),
            'avg_quality': np.mean(image_qualities) if image_qualities else 0,
            'std_quality': np.std(image_qualities) if image_qualities else 0,
            'generation_times': generation_times,
            'image_qualities': image_qualities
        }
        
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per image: {result['avg_time_per_image']:.3f}s")
        print(f"  Average quality: {result['avg_quality']:.4f}")
        
        return result
    
    def run_optimized_test(self, test_num: int = 20):
        """Run optimized test of all methods"""
        
        print("🚀 Starting Optimized Hessian Testing")
        print("="*60)
        print(f"Test configuration:")
        print(f"  - Model: {self.model_id}")
        print(f"  - Device: {self.device}")
        print(f"  - Test samples: {test_num}")
        print(f"  - Inference steps: 20")
        print("="*60)
        
        # Test each method
        methods = [
            ('original', self.test_original_lml),
            ('lightweight', self.test_lightweight_hessian),
            ('hessian_free_fast', self.test_fast_hessian_free)
        ]
        
        for method_name, test_func in methods:
            try:
                result = test_func(test_num)
                self.results[method_name] = result
            except Exception as e:
                print(f"❌ Error testing {method_name}: {e}")
                self.results[method_name] = {
                    'method': method_name,
                    'error': str(e),
                    'success': False
                }
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        
        print(f"\n{'='*60}")
        print("OPTIMIZED COMPARISON REPORT")
        print(f"{'='*60}")
        
        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not successful_results:
            print("❌ No successful results to compare")
            return
        
        # Performance comparison
        print(f"\n📊 PERFORMANCE COMPARISON")
        print(f"{'Method':<20} {'Time (s)':<12} {'Quality':<12} {'Speedup':<12} {'Efficiency':<12}")
        print("-" * 80)
        
        baseline_time = min(r['avg_time_per_image'] for r in successful_results.values())
        
        for method_name, result in successful_results.items():
            speedup = baseline_time / result['avg_time_per_image']
            efficiency = result['avg_quality'] / result['avg_time_per_image']  # Quality per second
            
            print(f"{method_name:<20} {result['avg_time_per_image']:<12.3f} {result['avg_quality']:<12.4f} {speedup:<12.2f}x {efficiency:<12.2f}")
        
        # Find best methods
        fastest_method = min(successful_results.keys(), 
                           key=lambda k: successful_results[k]['avg_time_per_image'])
        best_quality_method = max(successful_results.keys(), 
                                key=lambda k: successful_results[k]['avg_quality'])
        most_efficient_method = max(successful_results.keys(),
                                  key=lambda k: successful_results[k]['avg_quality'] / successful_results[k]['avg_time_per_image'])
        
        print(f"\n🎯 RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"🏆 Fastest method: {fastest_method}")
        print(f"   Time: {successful_results[fastest_method]['avg_time_per_image']:.3f}s per image")
        
        print(f"🎨 Best quality method: {best_quality_method}")
        print(f"   Quality: {successful_results[best_quality_method]['avg_quality']:.4f}")
        
        print(f"⚡ Most efficient method: {most_efficient_method}")
        print(f"   Efficiency: {successful_results[most_efficient_method]['avg_quality'] / successful_results[most_efficient_method]['avg_time_per_image']:.2f} quality/s")
        
        # Overall recommendation
        if most_efficient_method == fastest_method:
            print(f"\n✅ RECOMMENDATION: Use '{most_efficient_method}' - provides both speed and efficiency!")
        else:
            print(f"\n⚖️  TRADE-OFF ANALYSIS:")
            print(f"   - For maximum speed: Use '{fastest_method}'")
            print(f"   - For best quality: Use '{best_quality_method}'")
            print(f"   - For best efficiency: Use '{most_efficient_method}'")

def main():
    parser = argparse.ArgumentParser(description="Optimized Hessian testing")
    parser.add_argument('--model_id', type=str, default='./model/ddpm_ema_cifar10',
                        help='Path to the model')
    parser.add_argument('--test_num', type=int, default=20,
                        help='Number of test images to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Get absolute path
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_id = os.path.join(project_dir, args.model_id)
    
    # Run optimized test
    tester = OptimizedHessianTester(model_id, args.device)
    tester.run_optimized_test(args.test_num)

if __name__ == '__main__':
    main()
