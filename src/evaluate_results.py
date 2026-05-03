#!/usr/bin/env python3
"""
Evaluate pre-generated model outputs against the ObjectMover / Lightmove
ground truth using DreamSim / PSNR / LPIPS / DINO / CLIP.

This script does NOT run any model inference; it only compares images that
already exist on disk (`<sample>_result.png`) against the dataset's
`tar_input.jpg` ground truth, optionally cropping by source / target masks.

Layout expected:

    <gt_dataset_path>/
        <prefix>_001/
            tar_input.jpg
            src_mask_hr.png
            tar_box_mask.png
        ...

    <results_dir>/
        <prefix>_001_result.png
        <prefix>_002_result.png
        ...

`<prefix>` is auto-detected from the result filenames (e.g. `real_`,
`lightmove_`).
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_evaluator import MetricEvaluator


def crop_image_with_mask(mask_image: Image.Image, target_image: Image.Image, crop_channel: str = "r") -> Image.Image:
    """Crop `target_image` to the bounding box of valid pixels in `mask_image`."""
    if mask_image.mode != 'RGB':
        raise ValueError(f"Mask image must be in RGB mode, but got {mask_image.mode}")
    if target_image.mode != 'RGB':
        raise ValueError(f"Target image must be in RGB mode, but got {target_image.mode}")

    if mask_image.size != target_image.size:
        mask_image = mask_image.resize(target_image.size, Image.NEAREST)

    channels = mask_image.split()
    if crop_channel == "r":
        channel_array = np.array(channels[0])
    elif crop_channel == "g":
        channel_array = np.array(channels[1])
    else:
        raise ValueError(f"Invalid crop channel: {crop_channel}")

    valid_pixels = channel_array == 255
    if not np.any(valid_pixels):
        raise ValueError(f"No valid pixels found in the mask ({crop_channel.upper()} channel)")

    coords = np.argwhere(valid_pixels)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1

    bbox: Tuple[int, int, int, int] = (x_min, y_min, x_max, y_max)
    return target_image.crop(bbox)


class InferenceResultsEvaluator:
    """Evaluate pre-generated outputs against ground truth (no inference)."""

    def __init__(
        self,
        metric_evaluator: MetricEvaluator,
        results_dir: str,
        gt_dataset_path: str,
        evaluation_mode: str = "all",
        save_comparison: bool = False,
    ):
        self.metric_evaluator = metric_evaluator
        self.results_dir = Path(results_dir)
        self.gt_dataset_path = Path(gt_dataset_path)
        self.evaluation_mode = evaluation_mode
        self.save_comparison = save_comparison
        self.original_images = {'gt': {}, 'pred': {}}

    def _detect_sample_prefix(self) -> str:
        """Detect the sample prefix (e.g. 'real_', 'lightmove_') from result filenames."""
        for result_file in sorted(self.results_dir.glob("*_result.png")):
            name = result_file.stem.replace('_result', '')
            match = re.match(r'^([a-zA-Z_]+)', name)
            if match:
                return match.group(1)
        return 'real_'

    def _load_with_optional_crop(self, img_path: Path, sample_dir: Path) -> Tuple[np.ndarray, Image.Image]:
        img = Image.open(img_path).convert('RGB')

        if self.evaluation_mode in ['target_crop', 'source_crop']:
            if self.evaluation_mode == 'target_crop':
                mask_path = sample_dir / "tar_box_mask.png"
                channel = "g"
            else:
                mask_path = sample_dir / "src_mask_hr.png"
                channel = "r"

            if not mask_path.exists():
                print(f"Warning: Mask {mask_path.name} not found for {sample_dir.name}, skipping crop")
                return np.array(img), img

            mask_gray = Image.open(mask_path).convert('L')
            mask_np = np.array(mask_gray)
            mask_rgb_np = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
            if self.evaluation_mode == 'target_crop':
                mask_rgb_np[:, :, 1] = mask_np
            else:
                mask_rgb_np[:, :, 0] = mask_np
            mask_rgb = Image.fromarray(mask_rgb_np)
            cropped = crop_image_with_mask(mask_rgb, img, crop_channel=channel)
            return np.array(cropped), cropped
        return np.array(img), img

    def read_ground_truth_images(self) -> Dict[str, np.ndarray]:
        gt_images = {}
        sample_prefix = self._detect_sample_prefix()
        for sample_dir in sorted(self.gt_dataset_path.iterdir()):
            if not sample_dir.is_dir() or not sample_dir.name.startswith(sample_prefix):
                continue
            sample_name = sample_dir.name
            gt_img_path = sample_dir / "tar_input.jpg"
            if not gt_img_path.exists():
                print(f"Warning: Ground truth not found for {sample_name}")
                continue
            gt_img_np, gt_img_pil = self._load_with_optional_crop(gt_img_path, sample_dir)
            self.original_images['gt'][sample_name] = gt_img_pil
            gt_images[sample_name] = gt_img_np
        return gt_images

    def read_predicted_images(self) -> Dict[str, np.ndarray]:
        pred_images = {}
        sample_prefix = self._detect_sample_prefix()
        pattern = f"{sample_prefix}*_result.png"
        for result_file in sorted(self.results_dir.glob(pattern)):
            sample_name = result_file.stem.replace('_result', '')
            sample_dir = self.gt_dataset_path / sample_name
            pred_img_np, pred_img_pil = self._load_with_optional_crop(result_file, sample_dir)
            self.original_images['pred'][sample_name] = pred_img_pil
            pred_images[sample_name] = pred_img_np
        return pred_images

    def _read_full_images(self, source: str) -> Dict[str, np.ndarray]:
        """Read full (uncropped) images. `source` is 'gt' or 'pred'."""
        images = {}
        sample_prefix = self._detect_sample_prefix()
        if source == 'gt':
            for sample_dir in sorted(self.gt_dataset_path.iterdir()):
                if not sample_dir.is_dir() or not sample_dir.name.startswith(sample_prefix):
                    continue
                gt_img_path = sample_dir / "tar_input.jpg"
                if not gt_img_path.exists():
                    continue
                images[sample_dir.name] = np.array(Image.open(gt_img_path).convert('RGB'))
        else:
            pattern = f"{sample_prefix}*_result.png"
            for result_file in sorted(self.results_dir.glob(pattern)):
                sample_name = result_file.stem.replace('_result', '')
                images[sample_name] = np.array(Image.open(result_file).convert('RGB'))
        return images

    def evaluate(self) -> Dict:
        print(f"\nEvaluation Mode: {self.evaluation_mode}")
        print(f"Results Directory: {self.results_dir}")
        print(f"Ground Truth Directory: {self.gt_dataset_path}")

        print("\nReading ground truth images...")
        gt_images = self.read_ground_truth_images()
        print("Reading predicted images...")
        pred_images = self.read_predicted_images()

        gt_images_full = None
        pred_images_full = None
        if self.evaluation_mode in ['target_crop', 'source_crop']:
            print("Reading full ground truth images for PSNR...")
            gt_images_full = self._read_full_images('gt')
            print("Reading full predicted images for PSNR...")
            pred_images_full = self._read_full_images('pred')

        common_samples = set(gt_images.keys()) & set(pred_images.keys())
        missing_in_pred = set(gt_images.keys()) - set(pred_images.keys())
        missing_in_gt = set(pred_images.keys()) - set(gt_images.keys())
        if missing_in_pred:
            print(f"\nWarning: {len(missing_in_pred)} samples missing in predictions: "
                  f"{sorted(list(missing_in_pred))[:5]}...")
        if missing_in_gt:
            print(f"\nWarning: {len(missing_in_gt)} samples in predictions but not in ground truth: "
                  f"{sorted(list(missing_in_gt))[:5]}...")

        print(f"\nEvaluating {len(common_samples)} samples...")

        per_image_metrics = {}
        metrics_sum = defaultdict(float)
        successful_count = 0

        for sample_name in tqdm(sorted(common_samples), desc="Computing metrics"):
            gt_img = gt_images[sample_name]
            pred_img = pred_images[sample_name]

            if gt_img.shape != pred_img.shape:
                print(f"\nWarning: Size mismatch for {sample_name}: GT {gt_img.shape} vs Pred {pred_img.shape}")
                pred_img = np.array(
                    Image.fromarray(pred_img).resize((gt_img.shape[1], gt_img.shape[0]), Image.LANCZOS)
                )

            try:
                gt_img_full = gt_images_full.get(sample_name) if gt_images_full else None
                pred_img_full = pred_images_full.get(sample_name) if pred_images_full else None

                if gt_img_full is not None and pred_img_full is not None:
                    if gt_img_full.shape != pred_img_full.shape:
                        pred_img_full = np.array(
                            Image.fromarray(pred_img_full).resize(
                                (gt_img_full.shape[1], gt_img_full.shape[0]), Image.LANCZOS
                            )
                        )

                metrics = self.metric_evaluator.evaluate(gt_img, pred_img, gt_img_full, pred_img_full)
                per_image_metrics[sample_name] = metrics
                for metric_name, value in metrics.items():
                    metrics_sum[metric_name] += value
                successful_count += 1
            except Exception as e:
                print(f"\nError evaluating {sample_name}: {e}")
                continue

        average_metrics = {}
        if successful_count > 0:
            for metric_name, total_value in metrics_sum.items():
                average_metrics[metric_name] = total_value / successful_count

        return {
            'evaluation_mode': self.evaluation_mode,
            'checkpoint_name': self.results_dir.name,
            'total_samples': len(common_samples),
            'successful_evaluations': successful_count,
            'failed_evaluations': len(common_samples) - successful_count,
            'per_image_metrics': per_image_metrics,
            'average_metrics': average_metrics,
        }

    def save_comparison_images(self, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        common_samples = set(self.original_images['gt'].keys()) & set(self.original_images['pred'].keys())
        for sample_name in tqdm(sorted(common_samples), desc="Saving comparisons"):
            gt_img = self.original_images['gt'][sample_name]
            pred_img = self.original_images['pred'][sample_name]
            if gt_img.size != pred_img.size:
                pred_img = pred_img.resize(gt_img.size, Image.LANCZOS)
            total_width = gt_img.width + pred_img.width
            max_height = max(gt_img.height, pred_img.height)
            comparison = Image.new('RGB', (total_width, max_height))
            comparison.paste(gt_img, (0, 0))
            comparison.paste(pred_img, (gt_img.width, 0))
            comparison.save(output_path / f"{sample_name}_comparison.png")


def run_single_mode(args, config, metric_evaluator, eval_mode):
    evaluator = InferenceResultsEvaluator(
        metric_evaluator=metric_evaluator,
        results_dir=args.results_dir,
        gt_dataset_path=config.gt_dataset_path,
        evaluation_mode=eval_mode,
        save_comparison=config.save_comparison,
    )
    results = evaluator.evaluate()
    if config.save_comparison:
        comparison_dir = Path(config.comparison_output_dir) / Path(args.results_dir).name / eval_mode
        print(f"\nSaving comparison images to {comparison_dir}")
        evaluator.save_comparison_images(str(comparison_dir))
    return results


def run_crop_average(args, config, metric_evaluator):
    crop_results = {}
    for crop_mode in ['target_crop', 'source_crop']:
        print(f"\n  Evaluating {crop_mode}...")
        evaluator = InferenceResultsEvaluator(
            metric_evaluator=metric_evaluator,
            results_dir=args.results_dir,
            gt_dataset_path=config.gt_dataset_path,
            evaluation_mode=crop_mode,
            save_comparison=False,
        )
        crop_results[crop_mode] = evaluator.evaluate()

    target_results = crop_results['target_crop']
    source_results = crop_results['source_crop']
    common_samples = (set(target_results['per_image_metrics'].keys())
                      & set(source_results['per_image_metrics'].keys()))

    averaged_per_image = {}
    for sample_name in common_samples:
        target_metrics = target_results['per_image_metrics'][sample_name]
        source_metrics = source_results['per_image_metrics'][sample_name]
        averaged_per_image[sample_name] = {
            m: (target_metrics[m] + source_metrics[m]) / 2.0 for m in target_metrics.keys()
        }

    averaged_metrics = {
        m: (target_results['average_metrics'][m] + source_results['average_metrics'][m]) / 2.0
        for m in target_results['average_metrics'].keys()
    }

    return {
        'evaluation_mode': 'crop_average',
        'checkpoint_name': target_results['checkpoint_name'],
        'total_samples': len(common_samples),
        'successful_evaluations': len(common_samples),
        'failed_evaluations': 0,
        'per_image_metrics': averaged_per_image,
        'average_metrics': averaged_metrics,
        'component_results': {
            'target_crop': target_results,
            'source_crop': source_results,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pre-generated images on ObjMove-A or Lightmove-A"
    )
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing <sample>_result.png files")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml",
                        help="Path to evaluation configuration YAML")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: <results_dir>_evaluation_results.json)")
    parser.add_argument("--modes", type=str, nargs='+', default=None,
                        choices=['all', 'target_crop', 'source_crop', 'crop_average'],
                        help="Override evaluation modes from config")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    evaluation_modes = args.modes if args.modes is not None else config.evaluation_modes
    if isinstance(evaluation_modes, str):
        evaluation_modes = [evaluation_modes]

    metric_config = OmegaConf.load(config.metric_config_path)

    print("Initializing metric evaluator...")
    metric_evaluator = MetricEvaluator(metric_config)

    all_results = {}
    for eval_mode in evaluation_modes:
        print(f"\n{'='*80}\nRunning evaluation in '{eval_mode}' mode\n{'='*80}")
        if eval_mode == 'crop_average':
            results = run_crop_average(args, config, metric_evaluator)
        else:
            results = run_single_mode(args, config, metric_evaluator, eval_mode)

        all_results[eval_mode] = results

        print(f"\n{'='*80}")
        if eval_mode == 'crop_average':
            print(f"Results Summary for '{eval_mode}' mode (averaged from target_crop + source_crop):")
        else:
            print(f"Results Summary for '{eval_mode}' mode:")
        print('='*80)
        print(f"Successful evaluations: {results['successful_evaluations']}/{results['total_samples']}")
        print("\nAverage Metrics:")
        for metric_name, value in results['average_metrics'].items():
            print(f"  {metric_name}: {value:.6f}")

    if args.output is None:
        results_dir_name = Path(args.results_dir).name
        output_path = Path(args.results_dir).parent / f"{results_dir_name}_evaluation_results.json"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}\nEvaluation results saved to: {output_path}\n{'='*80}\n")


if __name__ == "__main__":
    main()
