"""
Image utilities for SmolVLM trainers.

Provides shared image preprocessing functions used across SFT and DPO trainers.
Features:
- Image resizing with fallback chain
- Caching of resized images to avoid re-processing
"""

import logging
import hashlib
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

# SmolVLM2-256M patch size
PATCH_SIZE = 16

# Cache directory for resized images
IMAGE_CACHE_DIR = Path(__file__).parent.parent / "datasets" / "image_cache"
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_image_cache_key(image_path: str, target_width: int, target_height: int) -> str:
    """Generate a cache key for the resized image based on path and target dimensions."""
    key_str = f"{image_path}_{target_width}x{target_height}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_image(cache_key: str) -> Image.Image:
    """Load cached image if it exists."""
    cache_path = IMAGE_CACHE_DIR / f"{cache_key}.png"
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load cached image: {e}")
    return None


def save_to_cache(image: Image.Image, cache_key: str):
    """Save resized image to cache."""
    cache_path = IMAGE_CACHE_DIR / f"{cache_key}.png"
    try:
        image.save(cache_path, "PNG")
    except Exception as e:
        logger.warning(f"Failed to cache image: {e}")


def cleanup_image_cache():
    """Clean up the image cache directory."""
    import shutil
    if IMAGE_CACHE_DIR.exists():
        try:
            size = sum(f.stat().st_size for f in IMAGE_CACHE_DIR.glob('*.png') if f.is_file())
            shutil.rmtree(IMAGE_CACHE_DIR)
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleaned image cache ({size / 1e9:.2f} GB)")
            return size
        except Exception as e:
            logger.warning(f"Failed to clean image cache: {e}")
    return 0


def round_to_patch_size(dim: int, patch_size: int = PATCH_SIZE) -> int:
    """Round dimension to nearest multiple of patch_size."""
    return max(patch_size, (dim // patch_size) * patch_size)


def prepare_image_with_fallback(image: Image.Image, image_path: str = None, force_patch_divisible: bool = True, use_cache: bool = True) -> Image.Image:
    """
    Prepare image for SmolVLM processor with fallback resize chain and caching.

    Strategy:
    - Images <= 1920px longest edge: Keep size but ensure divisible by patch_size
    - Images > 1920px: Resize to 1920px (paper recommendation for 256M/500M)
    - If resize fails, try 1024px, then 512px
    - Always ensure final dimensions are divisible by patch_size (16) for DPO compatibility
    - Caches resized images to avoid re-processing on subsequent runs

    The SmolVLM processor can handle images up to ~2048px, but the paper
    recommends 1920px longest edge for 256M/500M models during evaluation.

    Args:
        image: PIL Image to prepare
        image_path: Optional path for logging and caching purposes
        force_patch_divisible: If True, ensure dimensions are divisible by patch_size (16)
        use_cache: If True, use image cache for resized images

    Returns:
        RGB-converted image with dimensions divisible by patch_size
    """
    # Always convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Get original dimensions
    orig_width, orig_height = image.size
    longest_edge = max(orig_width, orig_height)

    # Determine target dimensions
    if longest_edge <= 1920:
        # Keep original size but ensure divisible by patch_size
        new_width = orig_width
        new_height = orig_height
    else:
        # Try resize fallback chain for large images
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
                break
        else:
            # Fallback to original if nothing matched
            new_width = orig_width
            new_height = orig_height

    # Ensure dimensions are divisible by patch_size (required for DPO training)
    if force_patch_divisible:
        new_width = round_to_patch_size(new_width)
        new_height = round_to_patch_size(new_height)

    # Only resize if dimensions changed
    if new_width != orig_width or new_height != orig_height:
        # Check cache first if we have a path
        cache_key = None
        if use_cache and image_path:
            cache_key = get_image_cache_key(image_path, new_width, new_height)
            cached = get_cached_image(cache_key)
            if cached is not None:
                logger.debug(f"Loaded cached resized image ({image_path})")
                return cached

        # Resize the image
        try:
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            path_info = f" ({image_path})" if image_path else ""
            logger.info(f"Resized image{path_info} from {orig_width}x{orig_height} to {new_width}x{new_height}")

            # Save to cache
            if use_cache and cache_key:
                save_to_cache(resized, cache_key)

            return resized
        except Exception as e:
            logger.warning(f"Failed to resize image: {e}")
            # Last resort: resize to 512x512 (guaranteed to work)
            try:
                resized = image.resize((512, 512), Image.Resampling.LANCZOS)
                logger.warning(f"Fallback resize to 512x512")
                return resized
            except Exception as e2:
                logger.error(f"All resize attempts failed: {e2}")
                return image

    return image
