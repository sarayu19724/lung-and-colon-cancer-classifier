"""The same image decoding and resizing is used for training and inference."""
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def read_image(source):
    if isinstance(source, bytes):
        if len(source) > MAX_UPLOAD_BYTES:
            raise ValueError("Choose an image smaller than 10 MB.")
        source = BytesIO(source)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Use a JPEG or PNG image.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 20 megapixel limit.")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Use a single-frame image.")
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("This image cannot be decoded safely. Try another JPEG or PNG.") from error


def image_array(source, size=224):
    image = read_image(source)
    # EfficientNet contains its own rescaling. Do not divide these pixels by 255.
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
