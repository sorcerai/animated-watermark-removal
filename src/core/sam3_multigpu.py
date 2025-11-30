#!/usr/bin/env python3
"""
SAM3 Multi-GPU Support for Watermark Detection

Splits SAM3 across multiple GPUs:
- Vision backbone on primary GPU (larger VRAM - e.g., RTX 3090)
- Language backbone on secondary GPU (smaller VRAM - e.g., RTX 3070)

This reduces VRAM pressure on the primary GPU by offloading text encoding.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor" / "sam3"))


class SAM3MultiGPUBackbone(nn.Module):
    """
    Multi-GPU wrapper for SAM3VLBackbone.

    Runs vision backbone on one GPU and language backbone on another,
    transferring tensors as needed.
    """

    def __init__(
        self,
        original_backbone,
        vision_device: str = "cuda:0",
        language_device: str = "cuda:1",
    ):
        """
        Initialize multi-GPU wrapper.

        Args:
            original_backbone: The SAM3VLBackbone instance
            vision_device: Device for vision backbone (should have more VRAM)
            language_device: Device for language backbone
        """
        super().__init__()
        self.vision_device = torch.device(vision_device)
        self.language_device = torch.device(language_device)

        # Move backbones to their respective devices
        self.vision_backbone = original_backbone.vision_backbone.to(self.vision_device)
        self.language_backbone = original_backbone.language_backbone.to(self.language_device)

        # Copy other attributes
        self.scalp = original_backbone.scalp
        self.act_ckpt_whole_vision_backbone = original_backbone.act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = original_backbone.act_ckpt_whole_language_backbone

        logger.info(f"SAM3 Multi-GPU initialized:")
        logger.info(f"  Vision backbone on {self.vision_device}")
        logger.info(f"  Language backbone on {self.language_device}")

    def forward(
        self,
        samples: torch.Tensor,
        captions: List[str],
        input_boxes: Optional[torch.Tensor] = None,
        additional_text: Optional[List[str]] = None,
    ):
        """
        Forward pass with multi-GPU support.

        Image features computed on vision_device, text features on language_device,
        then text features transferred to vision_device for downstream fusion.
        """
        # Run vision on its device
        samples_vision = samples.to(self.vision_device)
        output = self.forward_image(samples_vision)

        # Run text on its device (separate from vision)
        text_output = self.forward_text(
            captions, input_boxes, additional_text,
            device=self.language_device
        )

        # Transfer text outputs to vision device
        for key in text_output:
            if isinstance(text_output[key], torch.Tensor):
                text_output[key] = text_output[key].to(self.vision_device)

        output.update(text_output)
        return output

    def forward_image(self, samples: torch.Tensor):
        """Forward through vision backbone."""
        samples = samples.to(self.vision_device)

        # Forward through backbone
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(
            samples
        )

        if self.scalp > 0:
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )

        sam2_output = None
        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        output = {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }

        return output

    def forward_text(
        self,
        captions,
        input_boxes=None,
        additional_text=None,
        device=None
    ):
        """Forward through language backbone on its dedicated device.

        Note: The `device` parameter is ignored - we always use self.language_device
        to ensure the text encoder runs on the correct GPU.
        """
        from copy import copy
        from torch.nn.attention import sdpa_kernel, SDPBackend

        # Always use language_device, ignore passed device parameter
        device = self.language_device
        output = {}

        text_to_encode = copy(captions)
        if additional_text is not None:
            text_to_encode += additional_text

        sdpa_context = sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        )

        with sdpa_context:
            text_attention_mask, text_memory, text_embeds = self.language_backbone(
                text_to_encode, input_boxes, device=device
            )

        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text):]
            output["additional_text_mask"] = text_attention_mask[-len(additional_text):]

        text_memory = text_memory[:, :len(captions)]
        text_attention_mask = text_attention_mask[:len(captions)]
        text_embeds = text_embeds[:, :len(captions)]

        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds

        # Transfer all outputs to vision device for downstream processing
        for key in output:
            if isinstance(output[key], torch.Tensor):
                output[key] = output[key].to(self.vision_device)

        return output


def build_sam3_multigpu(
    checkpoint_path: Union[str, Path],
    bpe_path: Optional[Union[str, Path]] = None,
    vision_device: str = "cuda:0",
    language_device: str = "cuda:1",
    eval_mode: bool = True,
):
    """
    Build SAM3 image model with multi-GPU support.

    Args:
        checkpoint_path: Path to sam3.pt checkpoint
        bpe_path: Path to BPE vocab file (optional)
        vision_device: Device for vision backbone (needs more VRAM)
        language_device: Device for language backbone
        eval_mode: Set model to eval mode

    Returns:
        Sam3Image model with multi-GPU backbone
    """
    import os
    import sam3
    from sam3.model_builder import (
        _create_vision_backbone,
        _create_text_encoder,
        _create_vl_backbone,
        _create_sam3_transformer,
        _create_dot_product_scoring,
        _create_segmentation_head,
        _create_geometry_encoder,
        _load_checkpoint,
    )
    from sam3.model.sam3_image import Sam3Image

    checkpoint_path = Path(checkpoint_path)

    if bpe_path is None:
        sam3_root = Path(sam3.__file__).parent.parent
        bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    logger.info("Building SAM3 with multi-GPU support...")
    logger.info(f"  Vision device: {vision_device}")
    logger.info(f"  Language device: {language_device}")

    # Create components on CPU first
    logger.info("Creating vision encoder...")
    vision_encoder = _create_vision_backbone(compile_mode=None, enable_inst_interactivity=False)

    logger.info("Creating text encoder...")
    text_encoder = _create_text_encoder(str(bpe_path))

    logger.info("Creating VL backbone...")
    original_backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Wrap with multi-GPU support
    logger.info("Wrapping backbone for multi-GPU...")
    multigpu_backbone = SAM3MultiGPUBackbone(
        original_backbone,
        vision_device=vision_device,
        language_device=language_device,
    )

    logger.info("Creating transformer...")
    transformer = _create_sam3_transformer()

    logger.info("Creating other components...")
    dot_prod_scoring = _create_dot_product_scoring()
    segmentation_head = _create_segmentation_head(compile_mode=None)
    geometry_encoder = _create_geometry_encoder()

    # Build model
    logger.info("Building Sam3Image model...")
    model = Sam3Image(
        backbone=multigpu_backbone,
        transformer=transformer,
        input_geometry_encoder=geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=None,
        matcher=None,
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    _load_checkpoint(model, str(checkpoint_path))

    # Move remaining components to vision device
    logger.info(f"Moving transformer and other components to {vision_device}...")
    model.transformer.to(vision_device)
    model.geometry_encoder.to(vision_device)
    model.segmentation_head.to(vision_device)
    model.dot_prod_scoring.to(vision_device)

    if eval_mode:
        model.eval()

    # Log memory usage
    torch.cuda.synchronize()
    mem_vision = torch.cuda.memory_allocated(int(vision_device.split(":")[-1])) / 1024**3
    mem_language = torch.cuda.memory_allocated(int(language_device.split(":")[-1])) / 1024**3
    logger.info(f"Memory usage:")
    logger.info(f"  {vision_device}: {mem_vision:.2f} GB")
    logger.info(f"  {language_device}: {mem_language:.2f} GB")

    return model


class SAM3TextSegmenterMultiGPU:
    """
    SAM3 Text Segmenter with Multi-GPU support.

    Like SAM3TextSegmenter but splits model across GPUs.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        bpe_path: Optional[Union[str, Path]] = None,
        vision_device: str = "cuda:0",
        language_device: str = "cuda:1",
        confidence_threshold: float = 0.5,
        use_fp16: bool = True
    ):
        """
        Initialize multi-GPU SAM3 segmenter.

        Args:
            checkpoint_path: Path to sam3.pt
            bpe_path: Path to BPE vocab
            vision_device: GPU for vision (larger VRAM)
            language_device: GPU for text (smaller VRAM)
            confidence_threshold: Detection threshold
            use_fp16: Use half precision
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.bpe_path = bpe_path
        self.vision_device = vision_device
        self.language_device = language_device
        self.confidence_threshold = confidence_threshold
        self.use_fp16 = use_fp16

        self.model = None
        self.processor = None
        self._current_image = None
        self._inference_state = None
        self._load_model()

    def _load_model(self):
        """Load multi-GPU SAM3 model."""
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor

        logger.info(f"Loading SAM3 Multi-GPU from {self.checkpoint_path}")

        # Enable TF32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Build multi-GPU model
        self.model = build_sam3_multigpu(
            checkpoint_path=self.checkpoint_path,
            bpe_path=self.bpe_path,
            vision_device=self.vision_device,
            language_device=self.language_device,
            eval_mode=True,
        )

        # Create processor
        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=self.confidence_threshold
        )

        logger.info("SAM3 Multi-GPU loaded successfully")

    def set_image(self, image):
        """Set image for segmentation."""
        import numpy as np
        from PIL import Image as PILImage

        self._current_image = image

        # Convert BGR to RGB if needed
        if isinstance(image, np.ndarray) and image.shape[-1] == 3:
            image_rgb = image[:, :, ::-1].copy()
        else:
            image_rgb = image

        # Convert to PIL
        pil_image = PILImage.fromarray(image_rgb)

        # Set image
        self._inference_state = self.processor.set_image(pil_image)

    def segment_from_text(self, prompts, dilate_px: int = 3, use_batched: bool = True):
        """Segment regions matching text prompts.

        Args:
            prompts: List of text prompts
            dilate_px: Pixels to dilate mask
            use_batched: If True, process all prompts in one forward pass (faster)
        """
        if use_batched:
            return self.segment_from_text_batched(prompts, dilate_px=dilate_px)

        # Legacy sequential processing (slower, 4x more forward passes)
        import numpy as np
        from .mask_utils import create_empty_mask, merge_masks, dilate_mask

        if self._current_image is None:
            raise RuntimeError("No image set. Call set_image() first.")

        if isinstance(prompts, str):
            prompts = [prompts]

        all_masks = []

        for prompt in prompts:
            self.processor.reset_all_prompts(self._inference_state)
            self._inference_state = self.processor.set_text_prompt(
                state=self._inference_state,
                prompt=prompt
            )

            masks = self._inference_state.get('masks')
            if masks is not None and masks.shape[0] > 0:
                masks_np = masks.squeeze(1).cpu().numpy()
                for mask_np in masks_np:
                    all_masks.append(mask_np)

        if not all_masks:
            return create_empty_mask(self._current_image.shape)

        merged = merge_masks([(m * 255).astype(np.uint8) for m in all_masks])

        if dilate_px > 0:
            merged = dilate_mask(merged, kernel_size=dilate_px * 2 + 1)

        return merged

    def segment_from_text_batched(self, prompts, dilate_px: int = 3):
        """Segment regions matching text prompts - BATCHED version.

        Processes all prompts in a single forward pass through text encoder
        and grounding, which is ~4x faster than sequential processing.
        """
        import numpy as np
        from torch.nn.functional import interpolate
        from .mask_utils import create_empty_mask, merge_masks, dilate_mask

        if self._current_image is None:
            raise RuntimeError("No image set. Call set_image() first.")

        if isinstance(prompts, str):
            prompts = [prompts]

        state = self._inference_state

        if "backbone_out" not in state:
            raise ValueError("Image not properly set")

        # Process ALL prompts in one forward pass through text encoder
        text_outputs = self.model.backbone.forward_text(prompts)
        state["backbone_out"].update(text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Run grounding once for all prompts
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.processor.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]

        if out_masks.shape[0] == 0:
            return create_empty_mask(self._current_image.shape)

        # Resize masks to original image size
        img_h = state["original_height"]
        img_w = state["original_width"]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        # Convert to numpy and merge
        masks_np = (out_masks.squeeze(1) > 0.5).cpu().numpy()

        all_masks = []
        for mask_np in masks_np:
            all_masks.append(mask_np)

        if not all_masks:
            return create_empty_mask(self._current_image.shape)

        merged = merge_masks([(m * 255).astype(np.uint8) for m in all_masks])

        if dilate_px > 0:
            merged = dilate_mask(merged, kernel_size=dilate_px * 2 + 1)

        return merged

    def segment_frame_with_text(self, frame, prompts=None, dilate_px: int = 3):
        """Full pipeline: detect and segment with text prompts."""
        if prompts is None:
            prompts = ["SORA", "watermark", "text", "logo", "username"]

        self.set_image(frame)
        return self.segment_from_text(prompts, dilate_px=dilate_px)

    def reset(self):
        """Reset internal state."""
        self._current_image = None
        self._inference_state = None

    def __repr__(self):
        return (
            f"SAM3TextSegmenterMultiGPU("
            f"vision={self.vision_device}, "
            f"language={self.language_device})"
        )
