from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from agentify.core.config import ImageConfig

logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: str, image_config: ImageConfig) -> str:
    """Open, resize, and encode an image as JPEG base64."""
    try:
        with Image.open(image_path) as img_pil:
            if img_pil.mode not in ("RGB", "L"):
                img_pil = img_pil.convert("RGB")

            max_side = image_config.max_side_px
            img_pil.thumbnail((max_side, max_side))

            buf = BytesIO()
            img_pil.save(
                buf,
                format="JPEG",
                quality=image_config.quality,
                optimize=True,
            )
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except FileNotFoundError:
        logger.error("Image file not found: %s", image_path)
        raise
    except Exception as exc:
        logger.error("Image processing error for %s: %s", image_path, exc, exc_info=True)
        raise


def build_user_content(
    user_input: str,
    *,
    image_config: ImageConfig,
    image_path: Optional[str] = None,
    image_detail_override: Optional[str] = None,
) -> Optional[Union[str, List[Dict[str, Any]]]]:
    """Build user message content for text-only or multimodal inputs."""
    has_text = bool(user_input and user_input.strip())
    has_image = bool(image_path)

    if not has_text and not has_image:
        return None

    if not has_image:
        return user_input

    b64_image_data = encode_image_to_base64(image_path, image_config)
    detail_level = image_detail_override or image_config.detail

    parts: List[Dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_image_data}",
                "detail": detail_level,
            },
        }
    ]
    if has_text:
        parts.append({"type": "text", "text": user_input})

    return parts
