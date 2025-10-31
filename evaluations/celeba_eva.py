#!/usr/bin/env python3
"""
Unified CelebA-HQ Sampling Script with Enhanced Logging

This script provides a unified interface for CelebA-HQ image generation using various
diffusion sampling algorithms. It combines enhanced features with comprehensive
evaluation metrics, flexible configuration, and unified logging system.
"""

import sys
import time
import torch
import os
import json
import argparse
import glob
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd



def safe_array_conversion(img):
    """Safely convert PIL Image to numpy array without deprecation warnings"""
    if isinstance(img, Image.Image):
        # Convert PIL Image to numpy array without copy parameter
        return np.asarray(img)
    else:
        return img

def calculate_color_score(images):
    """Calculate ColorS metric - Colorfulness score"""
    color_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate colorfulness using standard deviation of color channels
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

            # Calculate mean and standard deviation for each channel
            mean_r, std_r = np.mean(r), np.std(r)
            mean_g, std_g = np.mean(g), np.std(g)
            mean_b, std_b = np.mean(b), np.std(b)

            # Colorfulness metric (simplified version)
            colorfulness = np.sqrt(std_r**2 + std_g**2 + std_b**2)
            color_scores.append(colorfulness)

    return np.mean(color_scores) if color_scores else 0.0

def calculate_face_score(images):
    """Calculate FS metric - Face Score (simplified)"""
    face_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to grayscale for face detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Simple face quality metric based on edge detection and symmetry
        # This is a simplified version - in practice, you'd use a trained face quality model

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Symmetry (left-right)
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])

        if left_half.shape == right_half.shape:
            symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        else:
            symmetry = 0.5

        # Combined face score
        face_score = (edge_density * 0.3 + symmetry * 0.7) * 10  # Scale to match expected range
        face_scores.append(face_score)

    return np.mean(face_scores) if face_scores else 0.0

def calculate_df_iqa(images):
    """Calculate DFIQA metric - Deep Face Image Quality Assessment (simplified)"""
    df_iqa_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Calculate image quality metrics
        # Sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Contrast
        contrast = gray.std()

        # Brightness distribution
        brightness = gray.mean()

        # Combined quality score (simplified)
        quality_score = (laplacian_var / 1000.0 + contrast / 100.0 + brightness / 255.0) / 3.0
        df_iqa_scores.append(quality_score)

    return np.mean(df_iqa_scores) if df_iqa_scores else 0.0

def calculate_pic_score(images):
    """Calculate PicS metric - Picture Score (aesthetic quality)"""
    pic_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate aesthetic quality metrics

            # Color harmony (simplified)
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            color_balance = 1.0 - np.std([r.mean(), g.mean(), b.mean()]) / 255.0

            # Composition (rule of thirds approximation)
            h, w = img_array.shape[:2]
            center_region = img_array[h//3:2*h//3, w//3:2*w//3]
            composition_score = center_region.std() / img_array.std() if img_array.std() > 0 else 0.5

            # Overall aesthetic score
            aesthetic_score = (color_balance * 0.4 + composition_score * 0.6) * 25  # Scale to match expected range
            pic_scores.append(aesthetic_score)

    return np.mean(pic_scores) if pic_scores else 0.0

def calculate_eat_score(images):
    """Calculate EAT metric - Enhanced Aesthetic Test (simplified)"""
    eat_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate enhanced aesthetic metrics

            # Color diversity
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            color_diversity = unique_colors / (img_array.shape[0] * img_array.shape[1]) * 1000

            # Edge complexity
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_complexity = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]) * 100

            # Combined EAT score
            eat_score = (color_diversity * 0.3 + edge_complexity * 0.7) * 0.05  # Scale to match expected range
            eat_scores.append(eat_score)

    return np.mean(eat_scores) if eat_scores else 0.0

def calculate_laion_score(images):
    """Calculate Laion metric - LAION aesthetic score (simplified)"""
    laion_scores = []

    for img in images:
        img_array = safe_array_conversion(img)

        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Calculate LAION-style aesthetic metrics

            # Image clarity
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            clarity = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Color saturation
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1])

            # Overall aesthetic quality
            laion_score = (clarity / 1000.0 + saturation / 255.0) * 2.5  # Scale to match expected range
            laion_scores.append(laion_score)

    return np.mean(laion_scores) if laion_scores else 0.0

def evaluate_images(images, sampler_type):
    """Evaluate images using all metrics"""
    project.info(f"\n📊 Evaluating {len(images)} images for {sampler_type}...")

    # Calculate all metrics
    color_s = calculate_color_score(images)
    fs = calculate_face_score(images)
    df_iqa = calculate_df_iqa(images)
    pic_s = calculate_pic_score(images)
    eat = calculate_eat_score(images)
    laion = calculate_laion_score(images)

    results = {
        'sampler_type': sampler_type,
        'num_images': len(images),
        'ColorS': color_s,
        'FS': fs,
        'DFIQA': df_iqa,
        'PicS': pic_s,
        'EAT': eat,
        'Laion': laion
    }

    project.info(f"  ColorS: {color_s:.3f}")
    project.info(f"  FS: {fs:.3f}")
    project.info(f"  DFIQA: {df_iqa:.3f}")
    project.info(f"  PicS: {pic_s:.3f}")
    project.info(f"  EAT: {eat:.3f}")
    project.info(f"  Laion: {laion:.3f}")

    return results


def generate_comparison_table(results_dict):
    """Generate comparison table in the format shown in the image"""

    # Define the methods in order
    methods = ['DDIM [75]', 'PNDM [50]', 'DPM [51]', 'DPM++ [52]', 'UniPC [91]', 'HILDA(Ours)']

    # Map sampler types to method names
    sampler_to_method = {
        'ddim': 'DDIM [75]',
        'pndm': 'PNDM [50]',
        'dpm': 'DPM [51]',
        'dpm++': 'DPM++ [52]',
        'unipc': 'UniPC [91]',
        # 'dpm_lm': 'LML',
        # 'ddim_lm': 'LML',
        'hessian_free': 'HILDA(Ours)'
    }

    # Create table data
    table_data = []
    for method in methods:
        # Find corresponding results
        method_results = None
        for sampler_type, results in results_dict.items():
            if sampler_to_method.get(sampler_type) == method:
                method_results = results
                break

        if method_results:
            table_data.append({
                'Method': method,
                'ColorS': method_results['ColorS'],
                'FS': method_results['FS'],
                'DFIQA': method_results['DFIQA'],
                'PicS': method_results['PicS'],
                'EAT': method_results['EAT'],
                'Laion': method_results['Laion']
            })

    return table_data

def format_table_with_ranking(table_data):
    """Format table with best and second-best highlighting"""

    # Extract metrics for ranking
    metrics = ['ColorS', 'FS', 'DFIQA', 'PicS', 'EAT', 'Laion']

    # Find best and second-best for each metric
    rankings = {}
    for metric in metrics:
        values = [(i, row[metric]) for i, row in enumerate(table_data)]
        values.sort(key=lambda x: x[1], reverse=True)  # Higher is better

        if len(values) >= 2:
            best_idx = values[0][0]
            second_best_idx = values[1][0]
            rankings[metric] = {'best': best_idx, 'second': second_best_idx}

    # Create formatted table
    project.info("\n" + "="*120)
    project.info("Table 2. Comparison of different samplers on CelebA-HQ unconditional generation.")
    project.info("Best results are bolded and the second best results are underlined.")
    project.info("="*120)

    # Header
    header = f"{'Methods':<15} {'Colorful':<20} {'Face Quality':<25} {'Aesthetic':<30}"
    project.info(header)
    subheader = f"{'':15} {'ColorS(↑)':<10} {'FS(↑)':<10} {'DFIQA(↑)':<10} {'PicS(↑)':<10} {'EAT(↑)':<10} {'Laion(↑)':<10}"
    project.info(subheader)
    project.info("-" * 120)

    # Data rows
    for i, row in enumerate(table_data):
        method = row['Method']
        line = f"{method:<15}"

        for metric in metrics:
            value = row[metric]
            formatted_value = f"{value:.3f}"

            # Apply formatting
            if metric in rankings:
                if i == rankings[metric]['best']:
                    formatted_value = f"**{formatted_value}**"  # Bold
                elif i == rankings[metric]['second']:
                    formatted_value = f"_{formatted_value}_"    # Underline

            line += f" {formatted_value:<10}"

        project.info(line)

    project.info("="*120)
