# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from math import sqrt, ceil
from typing import Literal

import cv2
import numpy as np

from .morph_utils import morphu
from .slice_utils import slu


# Forked from https://github.com/bertsky/ocrd_cis/blob/5cf22f5baa093ffaf0049e3c9756094116273598/
class imageu:
    """
    Colelction of useful image utility operations
    """
    @staticmethod
    def show(img: np.ndarray, name: str = '', height: int | None = None) -> None:
        """
        Display an image using OpenCV's `imshow` function.
        Args:
            img: Input image to display.
            name: Window name. Defaults to ''.
        """
        if height:
            h, w = img.shape[:2]
            scale = height / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )
        cv2.imshow(name, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    @staticmethod
    def binary_objects(img: np.ndarray) -> list[tuple[slice, ...]]:
        """
        Find objects in a binary image using connected components.
        Args:
            img: Binary image (0 for background, 1+ for objects).
        Returns:
            List of slices representing each object.
        """
        labels, _ = morphu.label(img)
        return morphu.find_objects(labels)
    
    @staticmethod
    def estimate_glyph_scale(
        img: np.ndarray, 
        method: Literal['height', 'width', 'area', 'rms'] = 'rms',
        default: int = 42
    ) -> int:
        """
        Estimate glyph scale using connected components statistics.
        Args:
            img: Input image.
            method: Method to compute scale. Defaults to 'rms'.
            default: Default value if no valid objects are found. Defaults to 0.
        Returns:
            Estimated glyph scale in pixels.
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            img, 
            connectivity=8
        )
        if n <= 1:
            return default
        s: list[int] = []
        for i in range(1, n):  # skip background
            h: int = stats[i, cv2.CC_STAT_HEIGHT]
            w: int = stats[i, cv2.CC_STAT_WIDTH]
            if h < 3 or w < 2:  # filter noise + punctuation
                continue
            if method == 'height':
                s.append(h)
            elif method == 'width':
                s.append(w)
            elif method == 'area':
                s.append(h * w)
            else:
                s.append(int(sqrt(h * w)))
        values = np.array(s)
        if np.any(values):
            return ceil(np.median(values))
        return default
    
    def estimate_page_scale(img: np.ndarray, default: tuple[int, int] = (25, 25)) -> tuple[int, int]:
        """
        Estimates the median contour size from an image or region.
        Args:
            image: The binary image (may be masked) to calculate the scale from.
        Returns:
            The median contour size, calculated from the biggest 50% of the contours (for noise removal).
        """
        _, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        if len(stats) > 1:  # stats[0] is the background
            widths = sorted(stats[1:, 2], reverse=True)
            heights = sorted(stats[1:, 3], reverse=True)
            avg_width = np.median(widths[:ceil(len(widths) * 0.5)])
            avg_height = np.median(heights[:ceil(len(heights) * 0.5)])
            return ceil(avg_width), ceil(avg_height)
        return default

    @staticmethod
    def midrange(img: np.ndarray, frac: float = 0.5) -> float:
        """
        Compute the midrange of pixel values in an image.
        Args:
            img: Input image.
            frac: Scaling factor for the midrange. Defaults to 0.5.
        Returns:
            Midrange value.
        """
        return frac * (np.amin(img) + np.amax(img))
    
    @staticmethod
    def clip_polygon(img: np.ndarray, polygon: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        Clip an image to the bounds of a polygon.
        Args:
            img: Input image.
            polygon: Polygon coordinates (N x 2 array of points).
        Returns:
            A tuple of:
              - Clipped image.
              - x_offset: Offset in x direction.
              - y_offset: Offset in y direction.
        """
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        result = np.full_like(img, 0)
        result[mask == 255] = img[mask == 255]
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)
        return result[y_min:y_max + 1, x_min:x_max + 1], x_min, y_min
    
    @staticmethod
    def compute_boxmap(
        img: np.ndarray,
        scale: float | int,
        threshold: tuple[float | int, float | int] = (0.5, 4), 
        dtype: str = 'i'
    ) -> np.ndarray:
        """
        Generate a box map based on object sizes and scale.
        Args:
            img: Input image.
            scale: Reference scale (e.g., glyph size).
            threshold: Threshold range for object sizes. Defaults to (0.5, 4).
            dtype: Data type for the output array. Defaults to 'i'.
        Returns:
            Box map where 1 indicates valid objects.
        """
        objects = imageu.binary_objects(img)
        bysize = sorted(objects, key=slu.area)
        boxmap = np.zeros(img.shape, dtype)
        for o in reversed(bysize):
            if slu.area(o) ** 0.5 < threshold[0] * scale:  # only too small boxes (noise) from here on
                break
            if slu.area(o) ** 0.5 > threshold[1] * scale:  # ignore too large box
                continue
            boxmap[o] = 1
        return boxmap
