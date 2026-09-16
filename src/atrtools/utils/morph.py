# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools

import cv2
import numpy as np
from scipy.ndimage import measurements

from .slice import SliceUtils


# Forked from https://github.com/bertsky/ocrd_cis/blob/5cf22f5baa093ffaf0049e3c9756094116273598/
class MorphUtils:
    @staticmethod
    def _brick(size: tuple[int, int]) -> np.ndarray:
        """
        Create a binary brick (square kernel) of given size.
        Args:
            size: Dimensions of the brick (height, width).
        Returns:
            A 2D NumPy array of shape `size` filled with ones.
        """
        return np.ones(size, np.uint8)

    @staticmethod
    def label(image: np.ndarray, **kwargs) -> tuple[np.ndarray, int]:
        """
        Label connected components in a binary image. Implements the scipy.ndimage.measurements.label function using 
        OpenCV's connectedComponents for improved performance.
        Args:
            image: Input binary image (0 for background, 1+ for objects)
        Returns:
            A tuple of:
              - labels (np.ndarray): Labeled image where each object has a unique integer.
            - n (int): Number of objects found (excluding background)
        """
        # default connectivity in OpenCV: 8 (which is equivalent to...)
        # default connectivity in scikit-image: 2
        # connectivity=4 crashes (segfaults) OpenCV#21366
        n, labels = cv2.connectedComponents(image.astype(np.uint8))
        return labels, n - 1

    @staticmethod
    def correspondences(labels1: np.ndarray, labels2: np.ndarray, return_counts: bool = True) -> np.ndarray:
        """
        Find label correspondences between two labeled images.
        Args:
            labels1: First labeled image.
            labels2: Second labeled image.
            return_counts: Whether to return pixel counts. Defaults to True.
        Returns:
            Array of label correspondences. If `return_counts` is True, it contains (label1, label2, count). 
            Otherwise, it contains (label1, label2).
        """
        q = 100000
        assert np.amin(labels1) >= 0 and np.amin(labels2) >= 0
        assert np.amax(labels2) < q
        combo = labels1 * q + labels2
        result = np.unique(combo, return_counts=return_counts)
        if return_counts:
            result, counts = result
            result = np.array([result // q, result % q, counts])
        else:
            result = np.array([result // q, result % q])  # ty:ignore[unsupported-operator]
        return result

    @staticmethod
    def find_objects(image: np.ndarray, **kwargs) -> list[tuple[slice, ...]]:
        """
        Find objects (regions of interest) in a labeled image. Redefines the scipy.ndimage.measurements.find_objects 
        function to support a wider range of data types, ensuring consistency across different platforms.
        Args:
            image: Input image (can be labeled or binary).
        Returns:
            List of slices for each object. Each slice is a tuple of `slice` objects for the object's bounding box.
        """
        try:
            return measurements.find_objects(image, **kwargs)
        except Exception:  # noqa: BLE001, S110
            pass
        types = ['int32', 'uint32', 'int64', 'uint64', 'int16', 'uint16']
        for t in types:
            try:
                return measurements.find_objects(np.array(image, dtype=t), **kwargs)
            except Exception:  # noqa: BLE001, S110
                pass
        return measurements.find_objects(image, **kwargs)  # let it raise the same exception as before

    @staticmethod
    def select_regions(img: np.ndarray, f, min: float = 0, nbest: int = 100000) -> np.ndarray:
        """
        Select regions from a labeled binary image based on a scoring function.
        Args:
            img: Binary image (0 for background, 1+ for objects).
            f: Scoring function that takes a region and returns a score.
            min: Minimum score to consider a region. Defaults to 0.
            nbest: Number of best regions to keep. Defaults to 100000.
        Returns:
            Array of shape `binary.shape` with 1s where regions are kept.
        """
        if img.max() == 1:
            labels, _ = MorphUtils.label(img)
        else:
            labels = img.astype(np.uint8)
        objects = MorphUtils.find_objects(labels)
        scores = [f(o) for o in objects]
        best = np.argsort(scores)
        keep = np.zeros(len(objects) + 1, "i")
        if nbest > 0:
            for i in best[-nbest:]:
                if scores[i] <= min:
                    continue
                keep[i + 1] = 1
        return keep[labels]

    @staticmethod
    def rb_opening(img, size: tuple[int, int], origin: int = 0) -> np.ndarray:
        """
        Perform binary opening on an image using a rectangular kernel.
        Args:
            img: Input binary image.
            size: Size of the structuring element (height, width).
            origin: Origin of the structuring element. Defaults to 0.
        Returns:
            Output image after opening.
        """
        return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, MorphUtils._brick(size))

    @staticmethod
    def rb_closing(img: np.ndarray, size: tuple[int, int], origin: int = 0) -> np.ndarray:
        """
        Perform binary closing on an image using a rectangular kernel.
        Args:
            img: Input binary image.
            size: Size of the structuring element (height, width).
            origin: Origin of the structuring element. Defaults to 0.
        Returns:
            Output image after closing.
        """
        return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE, MorphUtils._brick(size))
    
    @staticmethod
    def rb_reconstruction(img: np.ndarray, mask: np.ndarray, step: int = 1, maxsteps: int | None = None) -> np.ndarray:
        """
        Perform morphological reconstruction using dilation and masking.
        Args:
            img: Input image.
            mask: Mask image to constrain reconstruction.
            step: Step size for structuring element. Defaults to 1.
            maxsteps: Maximum number of steps. Defaults to None.
        Returns:
            Reconstructed image.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, 
            (2 * step + 1, 2 * step + 1)
        )
        dilated = img.astype(np.uint8)
        while maxsteps is None or maxsteps > 0:
            dilated = cv2.dilate(src=dilated, kernel=kernel)
            cv2.bitwise_and(src1=dilated, src2=mask.astype(np.uint8), dst=dilated)
            if (img == dilated).all():  # did result change?
                return dilated
            if maxsteps:
                maxsteps -= step
        return dilated

    @staticmethod
    def r_dilation(img: np.ndarray, size: tuple[int, int], origin: int = 0) -> np.ndarray:
        """
        Perform dilation on an image using a rectangular kernel.
        Args:
            img: Input image.
            size: Size of the structuring element (height, width).
            origin: Origin of the structuring element. Defaults to 0.
        Returns:
            Output image after dilation.
        """
        return cv2.dilate(img.astype(np.uint8), MorphUtils._brick(size))

    @staticmethod
    def r_closing(img: np.ndarray, size: tuple[int, int], origin: int = 0) -> np.ndarray:
        """
        Perform closing on an image using a rectangular kernel.
        Args:
            img: Input image.
            size: Size of the structuring element (height, width).
            origin: Origin of the structuring element. Defaults to 0.
        Returns:
            Output image after closing.
        """
        return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE, MorphUtils._brick(size))

    @staticmethod
    def propagate_labels_simple(regions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Propagate labels from one image to another based on label correspondence.
        Args:
            regions: Source image with regions to propagate.
            labels: Target image with labels to assign.
        Returns:
            Labeled image with propagated labels.
        """
        rlabels, _ = MorphUtils.label(regions)
        cors = MorphUtils.correspondences(rlabels, labels, False)
        outputs = np.zeros(np.amax(rlabels) + 1, 'i')
        for o, i in cors.T:
            outputs[o] = i
        outputs[0] = 0
        return outputs[rlabels]
    
    @staticmethod
    def propagate_labels_majority(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Propagate labels using majority voting for overlapping regions.
        Args:
            img: Input image.
            labels: Target image with labels to assign.
        Returns:
            Labeled image with propagated labels.
        """
        rlabels, _ = MorphUtils.label(img)
        cors = MorphUtils.correspondences(rlabels, labels)
        outputs = np.zeros(np.amax(rlabels) + 1, 'i')
        counts = np.zeros(np.amax(rlabels) + 1, 'i')
        for rlabel, label_, count in cors.T:
            if not rlabel or not label_:  # ignore background correspondences
                continue
            if counts[rlabel] < count:
                outputs[rlabel] = label_
                counts[rlabel] = count
        outputs[0] = 0
        return outputs[rlabels]
    
    @staticmethod
    def propagate_labels(img: np.ndarray, labels: np.ndarray, conflict: int = 0) -> np.ndarray:
        """
        Propagate labels, marking conflicts with a given value.
        Args:
            img: Input image.
            labels: Target image with labels to assign.
            conflict: Value to use for conflicting regions. Defaults to 0.
        Returns:
            Labeled image with propagated labels.
        """
        rlabels, _ = MorphUtils.label(img)
        cors = MorphUtils.correspondences(rlabels, labels, False)
        outputs = np.zeros(np.amax(rlabels) + 1, 'i')
        oops = -(1 << 30)
        for o, i in cors.T:
            if outputs[o] != 0:
                outputs[o] = oops
            else:
                outputs[o] = i
        outputs[outputs == oops] = conflict
        outputs[0] = 0
        return outputs[rlabels]

    @staticmethod
    def spread_labels(labels: np.ndarray, maxdist: float = 9999999) -> np.ndarray:
        """
        Spread labels to the background based on distance transform.
        Args:
            labels: Input labels.
            maxdist: Maximum distance to propagate. Defaults to 9999999.
        Returns:
            Labeled image with spread labels.
        """
        if not labels.any():
            return labels
        distances, indexes = cv2.distanceTransformWithLabels(
            np.array(labels == 0, np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        spread = labels[np.where(labels > 0)][indexes - 1]
        spread *= distances < maxdist
        return spread

    @staticmethod
    def reading_order(seg: np.ndarray, rl: bool = False, bt: bool = False) -> np.ndarray:
        """
        Compute a new order for labeled objects based on their y and x centers.
        Args:
            seg: Labeled image.
            rl: Reverse order in x-direction. Defaults to False.
            bt: Reverse order in y-direction. Defaults to False.
        Returns:
            Array of shape `seg.shape` with new ordering.
        """
        segmap = np.zeros(np.amax(seg) + 1, 'i')
        objects = [(slice(0, 0), slice(0, 0))] + MorphUtils.find_objects(seg)
        if len(objects) <= 2:  # nothing to do
            segmap[1:] = 1
            return segmap

        def pos(f, obj):
            return np.array([f(x) if x else float("nan") for x in obj])

        ys = pos(SliceUtils.ycenter, objects)
        yorder = np.argsort(ys)[:: -1 if bt else 1]
        groups = [[yorder[0]]]
        for i, j in itertools.pairwise(yorder):
            oi = objects[i]
            oj = objects[j]
            if (
                oi 
                and oj 
                and SliceUtils.yoverlaps(oi, oj) 
                and (SliceUtils.ycenter_in(oi, oj) or SliceUtils.ycenter_in(oj, oi)) 
                and not any(SliceUtils.xoverlaps(oj, objects[k]) and SliceUtils.xoverlap_rel(oj, objects[k]) > 0.1 
                    for k in groups[-1])
            ):
                groups[-1].append(j)
            else:
                groups.append([j])
        rorder = []
        for group in groups:
            group = np.array(group)
            xs = pos(SliceUtils.xcenter, [objects[i] for i in group])
            xorder = np.argsort(xs)[:: -1 if rl else 1]
            rorder.extend(group[xorder])
        for i, j in enumerate(rorder):
            segmap[j] = i
        return segmap
