"""
Image utilities for SmolVLM trainers.

Provides shared image preprocessing functions used across SFT and DPO trainers.
"""

import logging
from PIL import Image

logger = logging.getLogger(__name__)


def prepare_image_with_fallback(image: Image.Image, image_path: str = None) -> Image.Image:
    """
    Prepare image for SmolVLM processor with fallback resize chain.

    Strategy:
    - Images <= 1920px longest edge: Return as-is, let processor handle
    - Images > 1920px: Resize to 1920px (paper recommendation for 256M/500M)
    - If resize fails, try 1024px, then 512px

    The SmolVLM processor can handle images up to ~2048px, but the paper
    recommends 1920px longest edge for 256M/500M models during evaluation.

    Args:
        image: PIL Image to prepare
        image_path: Optional path for logging purposes

    Returns:
        RGB-converted image, optionally resized if > 1920px
    """
    # Always convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Get original dimensions
    orig_width, orig_height = image.size
    longest_edge = max(orig_width, orig_height)

    # If image is already small enough, return as-is
    # Paper recommends 1920px longest edge for 256M/500M models
    if longest_edge <= 1920:
        return image

    # Try resize fallback chain for large images
    # Start with 1920 (paper recommendation), then 1024, then 512
    fallback_sizes = [1920, 1024, 512]

    for target_size in fallback_sizes:
        if longest_edge > target_size:
            # Calculate new dimensions preserving aspect ratio
            if orig_width > orig_height:
                new_width = target_size
                new_height = int(orig_height * (target_size / orig_width))
            else:
                new_height = target_size
                new_width = int(orig_width * (target_size / orig_height))

            try:
                resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                path_info = f" ({image_path})" if image_path else ""
                logger.info(f"Resized large image{path_info} from {orig_width}x{orig_height} to {new_width}x{new_height} (target: {target_size}px longest edge)")
                return resized
            except Exception as e:
                logger.warning(f"Failed to resize to {target_size}px: {e}, trying smaller size...")
                continue

    # If all resizes failed, return original (processor will handle or error)
    logger.warning(f"All resize attempts failed, returning original {orig_width}x{orig_height} image")
    return image
