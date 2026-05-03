"""
Metric evaluator for comparing pairs of RGB images.

Supported metrics: dreamsim, psnr, lpips, dino, clip.

This file is a self-contained copy of the evaluator used internally for the
ObjectMover / Lightmove benchmarks, with no inference / model-loading
dependencies.
"""

from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import numpy as np
import torch


class MetricEvaluator:
    def __init__(self, config):
        """
        Args:
            config (OmegaConf.DictConfig): Configuration with a `metrics` list.
        """
        self.metrics = []
        for metric_config in config.metrics:
            if metric_config.name == 'dreamsim':
                from dreamsim import dreamsim
                device = metric_config.get('device', 'cuda')
                model, preprocess = dreamsim(
                    pretrained=metric_config.get('pretrained', True),
                    device=device,
                )
                self.metrics.append({
                    'name': 'dreamsim',
                    'model': model,
                    'preprocess': preprocess,
                    'device': device,
                })
            elif metric_config.name == 'psnr':
                self.metrics.append({
                    'name': 'psnr',
                    'data_range': metric_config.get('data_range', 255),
                })
            elif metric_config.name == 'lpips':
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
                device = metric_config.get('device', 'cuda')
                lpips_metric = LearnedPerceptualImagePatchSimilarity(
                    net_type=metric_config.get('net_type', 'vgg')
                ).to(device)
                self.metrics.append({
                    'name': 'lpips',
                    'metric': lpips_metric,
                    'device': device,
                })
            elif metric_config.name == 'dino':
                from transformers import ViTImageProcessor, ViTModel
                device = metric_config.get('device', 'cuda')
                model_name = metric_config.get('model_name', 'facebook/dino-vits16')
                processor = ViTImageProcessor.from_pretrained(model_name)
                model = ViTModel.from_pretrained(model_name).to(device)
                self.metrics.append({
                    'name': 'dino',
                    'model': model,
                    'processor': processor,
                    'device': device,
                })
            elif metric_config.name == 'clip':
                import clip
                device = metric_config.get('device', 'cuda')
                model_name = metric_config.get('model_name', 'ViT-B/32')
                model, preprocess = clip.load(model_name, device=device)
                self.metrics.append({
                    'name': 'clip',
                    'model': model,
                    'preprocess': preprocess,
                    'device': device,
                })
            else:
                raise ValueError(f"Unsupported metric: {metric_config.name}")

    def evaluate(self, img1_np, img2_np, img1_full_np=None, img2_full_np=None):
        """
        Compute all configured metrics between two RGB uint8 images.

        Args:
            img1_np, img2_np: Cropped (or full) HxWx3 uint8 arrays. Used by
                every metric except optionally PSNR.
            img1_full_np, img2_full_np: Full (uncropped) versions used by
                PSNR when present. Lets PSNR stay on the full image even when
                other metrics evaluate on a crop.
        """
        if not isinstance(img1_np, np.ndarray) or not isinstance(img2_np, np.ndarray):
            raise ValueError("Inputs must be NumPy arrays.")
        if img1_np.dtype != np.uint8 or img2_np.dtype != np.uint8:
            raise ValueError("Inputs must be of type uint8.")
        if img1_np.shape != img2_np.shape:
            raise ValueError("Input images must have the same dimensions.")
        if len(img1_np.shape) != 3 or img1_np.shape[2] != 3:
            raise ValueError("Input images must be three-channel (RGB).")
        if not (0 <= img1_np).all() or not (img1_np <= 255).all():
            raise ValueError("Pixel values in img1_np must be in the range [0, 255].")
        if not (0 <= img2_np).all() or not (img2_np <= 255).all():
            raise ValueError("Pixel values in img2_np must be in the range [0, 255].")

        if img1_full_np is not None and img2_full_np is not None:
            if not isinstance(img1_full_np, np.ndarray) or not isinstance(img2_full_np, np.ndarray):
                raise ValueError("Full image inputs must be NumPy arrays.")
            if img1_full_np.dtype != np.uint8 or img2_full_np.dtype != np.uint8:
                raise ValueError("Full image inputs must be of type uint8.")
            if img1_full_np.shape != img2_full_np.shape:
                raise ValueError("Full image inputs must have the same dimensions.")
            if len(img1_full_np.shape) != 3 or img1_full_np.shape[2] != 3:
                raise ValueError("Full image inputs must be three-channel (RGB).")

        results = {}
        for metric in self.metrics:
            if metric['name'] == 'dreamsim':
                device = metric['device']
                model = metric['model']
                preprocess = metric['preprocess']
                img1 = Image.fromarray(img1_np.astype('uint8'), 'RGB')
                img2 = Image.fromarray(img2_np.astype('uint8'), 'RGB')
                img1_tensor = preprocess(img1).to(device)
                img2_tensor = preprocess(img2).to(device)
                with torch.no_grad():
                    distance = model(img1_tensor, img2_tensor)
                results['dreamsim'] = distance.item()
            elif metric['name'] == 'psnr':
                # PSNR uses the full images when provided, else falls back to
                # the (possibly cropped) inputs.
                psnr_img1 = img1_full_np if img1_full_np is not None else img1_np
                psnr_img2 = img2_full_np if img2_full_np is not None else img2_np
                score = psnr_metric(
                    psnr_img1,
                    psnr_img2,
                    data_range=metric['data_range'],
                )
                results['psnr'] = score
            elif metric['name'] == 'lpips':
                device = metric['device']
                img1_tensor = self._prepare_image_for_lpips(img1_np).to(device)
                img2_tensor = self._prepare_image_for_lpips(img2_np).to(device)
                lpips_metric = metric['metric']
                with torch.no_grad():
                    distance = lpips_metric(img1_tensor, img2_tensor)
                results['lpips'] = distance.item()
            elif metric['name'] == 'dino':
                device = metric['device']
                model = metric['model']
                processor = metric['processor']
                img1_pil = Image.fromarray(img1_np)
                img2_pil = Image.fromarray(img2_np)
                inputs = processor(images=[img1_pil, img2_pil], return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                similarity = torch.sum(features[0] * features[1]).item()
                results['dino'] = similarity
            elif metric['name'] == 'clip':
                device = metric['device']
                model, preprocess = metric['model'], metric['preprocess']
                logit_scale = model.logit_scale.exp()
                img1_input = preprocess(Image.fromarray(img1_np)).unsqueeze(0).to(device)
                img2_input = preprocess(Image.fromarray(img2_np)).unsqueeze(0).to(device)
                features1 = model.encode_image(img1_input)
                features1 = features1 / features1.norm(dim=1, keepdim=True).to(torch.float32)
                features2 = model.encode_image(img2_input)
                features2 = features2 / features2.norm(dim=1, keepdim=True).to(torch.float32)
                score = logit_scale * (features1 * features2).sum()
                results['clip'] = score.item()
        return results

    @staticmethod
    def _prepare_image_for_lpips(img_np):
        img_tensor = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1)
        img_tensor = img_tensor / 255.0
        img_tensor = (img_tensor * 2) - 1
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor


def resize_image_np_uint8(img_np_uint8, target_size):
    if not isinstance(img_np_uint8, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")
    if img_np_uint8.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8.")
    img_pil = Image.fromarray(img_np_uint8)
    img_resized_pil = img_pil.resize(target_size, Image.LANCZOS)
    return np.array(img_resized_pil, dtype=np.uint8)


if __name__ == "__main__":
    import sys
    config = OmegaConf.load(sys.argv[1] if len(sys.argv) > 1 else 'configs/metric_config.yaml')
    evaluator = MetricEvaluator(config)
    print("MetricEvaluator initialized with metrics:",
          [m['name'] for m in evaluator.metrics])
