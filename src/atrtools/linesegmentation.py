# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, combinations
from os import PathLike
from pathlib import Path
from typing import Literal

import click
import cv2
import networkx as nx
import numpy as np
from pypxml import PageElement, PageType, PageXML
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.ndimage import (
    center_of_mass,
    find_objects,
    gaussian_filter,
    maximum_filter,
    shift,
    uniform_filter,
)
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely import set_precision
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.validation import explain_validity
from skimage import draw

from atrtools.utils import (
    ClickUtils,
    ImageUtils,
    MorphUtils,
    NumpyUtils,
    PageUtils,
    SliceUtils,
)


logger: logging.Logger = logging.getLogger(__name__)


# Forked from https://github.com/bertsky/ocrd_cis/blob/5cf22f5baa093ffaf0049e3c9756094116273598/
class LineSegmentation:
    def __init__(self, spread: float = 2.4, threads: int = 1) -> None:
        """
        Args:
            spread: Distance in points (pt) from the foreground to project text line (or text region) labels 
                    into the background for polygonal contours; If zero, project half a scale/capheight. 
                    Defaults to 2.4.
            threads: Number of threads for parallel region computation. Defaults to 1.
        """
        self.spread = spread
        self.threads = max(1, threads)
        
    def process(self, img: PathLike | np.ndarray, xml: PathLike | PageXML) -> PageXML:
        """
        Compute baselines and polygons for annotated TextRegions in a PAGE-XML file.
        Args:
            img: Binary iNpUtilt image.
            xml: PageXML object containing annotated TextRegions.
        Raises:
            ValueError: Invalid iNpUtilts
        Returns:
            The iNpUtilt PageXML object containing the computed TextLines
        """
        if isinstance(img, PathLike):
            img: np.ndarray | None = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f'Could not open image at {img!s}')
        img: np.ndarray = ~img
        if len(np.unique(img)) > 2:
            raise ValueError('Image is not binary')
        
        self.shape: tuple[int, int] = img.shape
        if len(self.shape) != 2 or not self.shape[0] or not self.shape[0]:
            raise ValueError(f'Invalid image shape: {self.shape}')
        logger.info(f'Image dimensions: {self.shape[0]}x{self.shape[1]}')
        
        if isinstance(xml, PathLike):
            xml: PageXML = PageXML.open(str(xml))
        
        regions: list[PageElement] = [r for r in xml.regions if r.pagetype == PageType.TextRegion]
        if self.threads == 1:
            for region in regions:
                self.__process_region(img, region)
        else:
            with ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix='lineseg') as executor:
                for region in regions:
                    executor.submit(self.__process_region, img, region)
        return xml
    
    def __process_region(self, img: np.ndarray, region: PageElement) -> None:
        region_polygon: np.ndarray | None = PageUtils.element_to_polygon(region)
        if region_polygon is None:
            logger.warning(f'Could not find coordinates in {region.pagetype.value} {region["id"]}')
            return
        
        textlines: list[PageElement] = list(region.find_all(pagetype=PageType.TextLine))
        for textline in textlines:
            textline.delete()
        if textlines:
            logger.info(f'Removed {len(textlines)} lines in {region.pagetype.value} {region["id"]}')
        
        clipped_img, x_offset, y_offset = ImageUtils.clip_polygon(img, region_polygon)
        clipped_region = PageUtils.shift_polygon(region_polygon, -x_offset, -y_offset)
        
        if clipped_img.size == 0:
            logger.warning(f'Invalid region after clipping in {region.pagetype.value} {region["id"]}')
            return
        
        region_bin = np.array(clipped_img <= ImageUtils.midrange(clipped_img), bool)
        sep_bin = np.zeros_like(region_bin, bool)
        ignore_labels = np.zeros_like(region_bin, int)
    
        try:
            labels, baselines = self.__compute_segmentation(
                clipped_img,
                seps=(sep_bin + ignore_labels) > 0,
                spread_dist=round(self.spread / 1.0 * 300 / 72),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f'Error computing segmentation for {region.pagetype.value} {region["id"]}: {e}')
            return

        logger.info(f'Found {len(np.unique(labels)) - 1} text lines in {region.pagetype.value} {region["id"]}')
        region_mask = np.zeros_like(region_bin, bool)
        region_mask[draw.polygon(clipped_region[:, 1], clipped_region[:, 0], region_mask.shape)] = True

        # ensure the new line labels do not extrude from the region:
        line_labels = labels * region_mask
        
        # find contours around labels (can be non-contiguous):
        line_polys, _ = self.__masks2polygons(
            bg_labels=line_labels,
            fg_bin=region_bin,
            baselines=baselines,
            min_area=640,
        )

        lid = 0
        for llabel, poly, baseline in line_polys:
            line_polygon = np.array(poly, dtype=np.float32)
            line_polygon = PageUtils.shift_polygon(
                line_polygon, 
                x_offset, y_offset, 
                upper=self.shape[::-1]
            )
            line_polygon = self.__clip_line(line_polygon, region_polygon)
            if line_polygon is None:
                logger.info(f'Ignoring extant line contour for line label {llabel}')
                continue
            lid += 1
            lelement = region.create(PageType.TextLine, id=f'{region["id"]}_l{lid}')
            lelement.create(PageType.Coords, points=PageUtils.polygon_to_points(line_polygon))
            
            if baseline:
                line_baseline = np.array(baseline, dtype=np.float32)
                line_baseline = PageUtils.shift_polygon(
                    line_baseline, 
                    x_offset, y_offset, 
                    upper=self.shape[::-1]
                )
                
                # fix jumping baseline anomaly
                for i in range(1, len(line_baseline)):
                    if line_baseline[i, 0] < line_baseline[i - 1, 0] and len(line_baseline) > (i + 1):
                        line_baseline = line_baseline[i:]
                        break
                for i in range(len(line_baseline) - 2, 0, -1):
                    if line_baseline[i, 0] > line_baseline[i + 1, 0] and len(line_baseline) > (i + 1):
                        line_baseline = line_baseline[:i + 1]
                        break
                                
                lelement.create(PageType.Baseline, points=PageUtils.polygon_to_points(line_baseline))
    
    def __compute_gradmaps(
        self, 
        img: np.ndarray,
        scale: float,
        usegauss: bool = False, 
        vscale: float = 1.0, 
        hscale: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # use gradient filtering to find baselines. Default ocropy min,max scale filter: (0.5,4)
        # within regions/blocks, we will have remainders of (possibly rotated and chopped) text lines at the
        # region boundaries, which we want to regard as neighbouring independent/full lines (especially
        # during resegmentation) so we could use a smaller minimum threshold
        threshold = (0.5, 4)
        boxmap = ImageUtils.compute_boxmap(img, scale, threshold=threshold)
        cleaned = boxmap * img
        
        # find vertical edges
        if usegauss:  # this uses Gaussians
            grad = gaussian_filter(
                1.0 * cleaned,
                (vscale * 0.3 * scale, hscale * scale),
                order=(1, 0),
            ) 
        else:  # this uses non-Gaussian oriented filters
            grad = gaussian_filter(
                1.0 * cleaned,
                (max(4, vscale * 0.3 * scale), hscale * scale),
                order=(1, 0),
            )
            grad = uniform_filter(grad, (vscale, hscale * scale))
        bottom = NumpyUtils.norm_max((grad < 0) * (-grad))
        top = NumpyUtils.norm_max((grad > 0) * grad)
        return bottom, top, boxmap
    
    def __compute_line_seeds(
        self,
        img: np.ndarray,
        bottom: np.ndarray,
        top: np.ndarray,
        colseps: np.ndarray,
        scale: float,
        threshold: float = 0.2,
        vscale: float = 2.0,
        robust: bool = True
    ) -> np.ndarray:
        vrange = NumpyUtils.odd(vscale * scale)
        # find (more or less) horizontal lines along the maximum gradient,
        # where it is above (squared) threshold and not crossing columns:
        bmarked = (
            maximum_filter(
                bottom == maximum_filter(bottom, (vrange, 0)),  # mark position of maximum gradient every `vrange` pixels:
                (2, 2),  #  blur by 2 pixels, then retain only large gradients
            ) * (bottom > threshold * np.amax(bottom) * threshold) * (1 - colseps)
        )
        tmarked = (
            maximum_filter(
                top == maximum_filter(top, (vrange, 0)),  # mark position of maximum gradient every `vrange` pixels:
                (2, 2),  # blur by 2 pixels, then retain only large gradients:
            ) * (top > threshold * np.amax(top) * threshold / 2) * (1 - colseps)
        )
        if robust:
            bmarked = maximum_filter(bmarked, (1, NumpyUtils.odd(scale))) * (1 - colseps)
        tmarked = maximum_filter(tmarked, (1, NumpyUtils.odd(scale))) * (1 - colseps)

        seeds = np.zeros(img.shape, "i")
        delta = max(3, int(scale))
        for x in range(bmarked.shape[1]):
            # sort both kinds of mark from bottom to top (i.e. inverse y position)
            transitions = sorted(
                [(y, 1) for y in NumpyUtils.find(bmarked[:, x])]
                + [(y, 0) for y in NumpyUtils.find(tmarked[:, x])]
            )[::-1]
            if robust:
                ln = 0
                while ln < len(transitions):
                    y0, s0 = transitions[ln]
                    if s0:  # bmarked?
                        y1 = max(0, y0 - delta)  # project seed from bottom
                        if ln + 1 < len(transitions) and transitions[ln + 1][0] > y1:
                            y1 = transitions[ln + 1][0]  # fill with seed to next mark
                        seeds[y1:y0, x] = 1
                    else:  # tmarked?
                        y1 = y0 + delta  # project seed from top
                        if ln > 0 and transitions[ln - 1][0] < y1:
                            y1 = transitions[ln - 1][0]  # fill with seed to next mark
                        seeds[y0:y1, x] = 1
                    ln += 1
            else:
                transitions += [(0, 0)]
                for ln in range(len(transitions) - 1):
                    y0, s0 = transitions[ln]
                    if s0 == 0:
                        continue  # keep looking for next bottom
                    seeds[y0 - delta : y0, x] = 1  # project seed from bottom
                    y1, s1 = transitions[ln + 1]
                    if s1 == 0 and (y0 - y1) < 5 * scale:  # why 5?
                        # consistent next top?
                        seeds[y1:y0, x] = 1  # fill with seed completely
        if robust:
            # try to separate lines that already touch:
            seeds = MorphUtils.rb_opening(seeds, (NumpyUtils.odd(scale / 2), NumpyUtils.odd(scale)))
        else:
            # this will smear into neighbouring line components at as/descenders
            # (but horizontal consistency is now achieved by hmerge and spread):
            seeds = maximum_filter(seeds, (1, NumpyUtils.odd(1 +  scale)))
        # interrupt by column separators before labelling:
        seeds = seeds * (1 - colseps)
        seeds, _ = MorphUtils.label(seeds)
        return seeds
    
    def __hmerge_line_seeds(
        self,
        img: np.ndarray,
        seeds: np.ndarray,
        scale: int,
        threshold: float = 0.8,
        seps: np.ndarray | None = None,
    ) -> np.ndarray:
        # merge labels horizontally to avoid splitting lines at long whitespace (ensuring contiguous contours), but
        # ignore conflicts which affect only small fractions of either line 
        # (avoiding merges for small vertical overlap):
        labels = np.unique(seeds * (img > 0))  # without empty foreground
        labels = labels[labels > 0]  # without background
        seeds[~np.isin(seeds, labels, assume_unique=True)] = 0
        if len(labels) < 2:
            return seeds
        objects = find_objects(seeds)
        centers = center_of_mass(img, seeds, labels)
        relabel = np.arange(np.max(seeds) + 1, dtype=seeds.dtype)
        # FIXME: get incidence of y overlaps to avoid full inner loops
        logger.debug(f'Checking {len(labels)} non-empty line seeds for overlaps')

        def h_compatible(obj1, obj2, center1, center2):
            if not (obj2[0].start < center1[0] < obj2[0].stop):
                return False
            if not (obj1[0].start < center2[0] < obj1[0].stop):
                return False
            if obj2[1].start < center1[1] < obj2[1].stop:
                return False
            return not obj1[1].start < center2[1] < obj1[1].stop

        for label in labels:
            seed = seeds == label
            if not seed.any():
                continue
            
            # close to fill holes from underestimated scale
            seed = MorphUtils.rb_closing(seed, (scale, scale))
            if not seed.any():
                continue
            obj = find_objects(seed)[0]
            if obj is None:
                continue
            seed[obj[0], 0 : seed.shape[1]] = 1
            
            # get overlaps
            for label2 in labels:
                if label == label2 or relabel[label] == label2:
                    continue
                obj2 = objects[label2 - 1]
                if not obj2:
                    continue
                if not SliceUtils.yoverlaps(obj, obj2):
                    continue
                center = centers[labels.searchsorted(label)]
                bbox = objects[label - 1]
                if not all(
                    h_compatible(bbox, bbox2, center, center2)
                    for bbox2, center2 in [
                        (objects[i - 1], centers[labels.searchsorted(i)]) 
                        for i in np.nonzero(relabel == relabel[label2])[0]
                    ]
                ):
                    logger.debug(f'Ignoring h-overlap between {label} and {label2} (not mutually centric)')
                    continue
                seed2 = seeds == label2
                count = np.count_nonzero(seed2 * seed)
                total = np.count_nonzero(seed2)
                if count < threshold * total:
                    logger.debug(f'Ignoring h-overlap between {label} and {label2} (only {count} of {total})')
                    continue
                label1_y, label1_x = np.where(seeds == label)
                label2_y, label2_x = np.where(seed2)
                shared_y = np.intersect1d(label1_y, label2_y)
                gap = np.zeros_like(seed2, bool)
                for y in shared_y:
                    can_x_min = label2_x[label2_y == y][0]
                    can_x_max = label2_x[label2_y == y][-1]
                    new_x_min = label1_x[label1_y == y][0]
                    new_x_max = label1_x[label1_y == y][-1]
                    if (can_x_max < new_x_min) and (seps is None or not seps[y, can_x_max:new_x_min].any()):
                        gap[y, can_x_max:new_x_min] = True
                    if (new_x_max < can_x_min) and (seps is None or not seps[y, new_x_max:can_x_min].any()):
                        gap[y, new_x_max:can_x_min] = True
                if not gap.any() or gap.max(axis=1).sum() / len(shared_y) < 0.1:
                    logger.debug(f'Ignoring h-overlap between {label} and {label2} (blocked by seps)')
                    continue
                
                # find y with shortest gap
                gapwidth = gap.sum(axis=1)
                gapwidth[gapwidth == 0] = seed.shape[1]
                mingap = gapwidth < gapwidth.min() + 4
                
                # make contiguous
                mingap = mingap.nonzero()[0]
                gap[0 : mingap[0]] = False
                gap[mingap[-1] :] = False
                logger.debug(f'hmerging {label2} with {label}')
                
                seeds[gap] = label  # fill the horizontal background between both regions
                new_label = relabel[label]  # the new label could have been relabelled already
                relabel[label2] = new_label  # assign candidate to (new assignment for) label
                relabel[relabel == label2] = new_label  # re-assign labels already relabelled to candidate
                
        # apply re-assignments:
        seeds = relabel[seeds]
        return seeds
    
    def __compute_baselines(
        self, 
        bottom: np.ndarray, 
        top: np.ndarray, 
        linelabels: np.ndarray,
        scale: float,
        method: Literal['bottom', 'center', 'top'] = 'bottom'
    ) -> list[np.ndarray]:
        seeds = linelabels > 0
        
        # smooth bottom+top maps horizontally for centerline estimation
        bot = gaussian_filter(bottom, (scale * 0.25, scale), mode='constant')
        top = gaussian_filter(top, (scale * 0.25, scale), mode='constant')
        
        # idea: center is where bottom and top gradient meet in the middle
        # (but between top and bottom, not between bottom and top)
        # - calculation via numpy == or isclose is too fragile numerically:
        # clines = np.isclose(top, bottom, rtol=0.5) & (np.diff(top - bottom, axis=0, append=0) < 0)
        # - calculation via zero crossing of bop-bottom is more robust,
        #   but needs post-processing for lines with much larger height than scale
        # - calculation via peak gradient
        
        if method == 'center':
            blines = (np.diff(np.sign(top - bottom), axis=0, append=0) < 0) & seeds
        elif method == 'bottom':
            bot1d = np.diff(bot, axis=0, append=0)
            bot1d = np.diff(np.sign(bot1d), axis=0, append=0) < 0
            bot1d &= bot > 0
            blines = bot1d
            
        baselabels, _ = MorphUtils.label(blines)
        baseslices = [(slice(0, 0), slice(0, 0))] + MorphUtils.find_objects(baselabels)
        
        # if multiple labels per seed, ignore the ones above others (can happen due to mis-estimation of scale)
        corrs = MorphUtils.correspondences(linelabels, baselabels).T
        labelmap = {}
        for line in np.unique(linelabels):
            if not line:
                continue  # ignore bg line
            
            corrinds = corrs[:, 0] == line
            corrinds[corrs[:, 1] == 0] = False  # ignore bg baseline
            if not np.any(corrinds):
                continue
            
            corrinds = corrinds.nonzero()[0]
            if len(corrinds) == 1:
                labelmap.setdefault(line, []).append(corrs[corrinds[0], 1])
                continue
            
            nonoverlapping = ~np.eye(len(corrinds), dtype=bool)
            for i, indi in enumerate(corrinds[:-1]):
                baselabeli = corrs[indi, 1]
                baseslicei = baseslices[baselabeli]
                for j, indj in enumerate(corrinds[i + 1 :], i + 1):
                    baselabelj = corrs[indj, 1]
                    baseslicej = baseslices[baselabelj]
                    if SliceUtils.xoverlaps(baseslicei, baseslicej):
                        nonoverlapping[i, j] = False
                        nonoverlapping[j, i] = False

            # find all maximal cliques in the graph (i.e. all fully connected subgraphs)
            # and then pick the partition with the largest sum of pixels at its nodes
            def pathlen(path):
                return sum(corrs[corrinds[path], 2])  # noqa: B023

            corrinds = corrinds[max(nx.find_cliques(nx.Graph(nonoverlapping)), key=pathlen)]
            labelmap.setdefault(line, []).extend(corrs[corrinds, 1])

        basepoints = []
        for line in np.unique(linelabels):
            if line not in labelmap:
                continue
            
            linemask = linelabels == line
            points = []
            for label in labelmap[line]:
                points.extend(list(zip(*np.where((baselabels == label) & linemask))))
            basepoints.append(points)
        return basepoints  # ty: ignore[invalid-return-type]
        
    def __compute_segmentation(
        self, 
        img: np.ndarray, 
        seps: np.ndarray | None = None,
        spread_dist: float | None = None,
        rl: bool = False,
        bt: bool = False,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        logger.debug('Estimating glyph scale')
        scale = ImageUtils.estimate_glyph_scale(img, 'rms', 42)
        if seps is not None and not seps.all():
            # suppress separators/images for line estimation (unless it encompasses the full image for some reason)
            img = (1 - seps) * img
            
        logger.debug('Computing gradient map')
        bottom, top, _ = self.__compute_gradmaps(img, scale, False)
        sepmask = np.zeros(img.shape, np.uint8)
        
        logger.debug('Computing line seeds')
        seeds = self.__compute_line_seeds(img, bottom, top, sepmask, scale)
        seeds = self.__hmerge_line_seeds(img, seeds, scale, seps=sepmask)
        
        # spread labels from seeds to bg, but watch fg, voting for majority on bg conflicts, 
        # but splitting on seed conflicts
        logger.debug('Spreading seed labels')
        llabels = MorphUtils.propagate_labels_majority(img, seeds)
        llabels2 = MorphUtils.propagate_labels(img, seeds, conflict=0)
        conflicts = llabels > llabels2
        llabels = np.where(conflicts, seeds, llabels)
        
        # capture diacritics (isolated components above seeds)
        seeds2 = shift(seeds, (-scale, 0), order=0, prefilter=False)
        seeds2 = np.where(seeds, seeds, seeds2)
        llabels2 = MorphUtils.propagate_labels_simple(img, seeds2)
        llabels = np.where(llabels, llabels, llabels2)
        
        # (protect sepmask as a temporary label)
        seplabel = np.max(seeds) + 1
        llabels[sepmask > 0] = seplabel
        spread = MorphUtils.spread_labels(llabels, maxdist=spread_dist or scale / 2)
        llabels2 = MorphUtils.propagate_labels_majority(img, spread)
        llabels = np.where(seeds, seeds, llabels2)
        llabels[sepmask > 0] = seplabel
        llabels = MorphUtils.spread_labels(llabels, maxdist=spread_dist or scale / 2)
        llabels[llabels == seplabel] = 0

        logger.debug('Sorting labels by reading order')
        try:
            llabels = MorphUtils.reading_order(llabels, rl, bt)[llabels]
        except IndexError as exc:
            logger.warning(f"Could not sort lines: {exc}")

        # return segmentation
        blines = self.__compute_baselines(bottom, top, llabels, scale)
        return llabels, blines
    
    def __make_valid(self, polygon: Polygon) -> Polygon:
        points = list(polygon.exterior.coords)
        for split in range(1, len(points)):
            if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
                break
            
            # simplification may not be possible (at all) due to ordering in that case, try another starting point
            polygon = Polygon(points[-split:] + points[:-split])
        for tolerance in range(int(polygon.area)):
            if polygon.is_valid:
                break
            
            # simplification may require a larger tolerance
            polygon = polygon.simplify(tolerance + 1)
        return polygon
    
    def __join_baselines(self, baselines: list[LineString]) -> LineString | None:
        lines = []
        for baseline in baselines:
            if baseline.is_empty or baseline.geom_type in ['Point', 'MultiPoint']:
                continue
            elif baseline.geom_type == 'MultiLineString':
                lines.extend(baseline.geoms)
            elif baseline.geom_type == 'LineString':
                lines.append(baseline)
            elif baseline.geom_type == 'GeometryCollection':
                for geom in baseline.geoms:
                    if geom.geom_type == 'LineString':
                        lines.append(geom)
                    elif geom.geom_type == 'MultiLineString':
                        lines.extend(geom)
                    else:
                        logger.debug(f'Ignoring baseline subtype {geom.geom_type}')
            else:
                logger.debug(f'Ignoring baseline type {baseline.geom_type}')
        nlines = len(lines)
        if nlines == 0:
            return None
        elif nlines == 1:
            return lines[0]
        
        # find min-dist path through all lines (travelling salesman)
        pairs = combinations(range(nlines), 2)
        dists = np.eye(nlines, dtype=float)
        for i, j in pairs:
            dist = lines[i].distance(lines[j])
            dist = max(dist, 1e-5)  # if pair merely touches, we still need to get an edge
            dists[i, j] = dist
            dists[j, i] = dist
        dists = minimum_spanning_tree(dists, overwrite=True)
        assert dists.nonzero()[0].size, dists
        
        # get path
        chains = []
        for prevl, nextl in zip(*dists.nonzero()):
            foundchains = []
            for c in chains:
                if c[0] == prevl:
                    found = c, 0, nextl
                elif c[0] == nextl:
                    found = c, 0, prevl
                elif c[-1] == prevl:
                    found = c, -1, nextl
                elif c[-1] == nextl:
                    found = c, -1, prevl
                else:
                    continue
                foundchains.append(found)
            if len(foundchains):
                assert len(foundchains) <= 2, foundchains
                chain, pos, node = foundchains.pop()
                if len(foundchains):
                    otherchain, otherpos, othernode = foundchains.pop()
                    assert node != othernode
                    assert chain[pos] == othernode
                    assert otherchain[otherpos] == node
                    if pos < 0 and otherpos < 0:
                        chain.extend(reversed(otherchain))
                        chains.remove(otherchain)
                    elif pos < 0 and otherpos == 0:
                        chain.extend(otherchain)
                        chains.remove(otherchain)
                    elif pos == 0 and otherpos == 0:
                        otherchain.extend(reversed(chain))
                        chains.remove(chain)
                    elif pos == 0 and otherpos < 0:
                        otherchain.extend(chain)
                        chains.remove(chain)
                elif pos < 0:
                    chain.append(node)
                else:
                    chain.insert(0, node)
            else:
                chains.append([prevl, nextl])
    
        if len(chains) > 1:
            logger.debug('Baseline merge impossible (no spanning tree)')
            return None
        
        assert len(chains) == 1, chains
        assert len(chains[0]) == nlines, chains[0]
        
        # get points
        path = chains[0]
        coords = []
        for node in path:
            line = lines[node]
            coords.extend(line.normalize().coords)
        result = LineString(coords)
        
        if result.is_empty:
            logger.debug('Baseline merge is empty')
            return None
        
        assert result.geom_type == "LineString", result.wkt
        
        result = set_precision(result, 1.0)
        if result.geom_type != "LineString" or not result.is_valid:
            result = LineString(np.round(line.coords))
        return result
    
    def __masks2polygons(
        self,
        bg_labels: np.ndarray,
        fg_bin: np.ndarray,
        baselines: list[np.ndarray] | None = None,
        min_area: int | None = None,
    ) -> tuple[list[Polygon], np.ndarray]:
        # find sharp baseline
        if baselines is not None:
            baselines: list[LineString] = [
                LineString(sorted([p[::-1] for p in line], key=lambda xy: xy[0])).simplify(5)
                for line in baselines
                if len(line) >= 2
            ]
        results = []
        result_labels = np.zeros_like(bg_labels, dtype=bg_labels.dtype)
        for label in np.unique(bg_labels):
            if not label:
                # ignore if background
                continue
            bg_mask = np.array(bg_labels == label, bool)
            if not np.count_nonzero(bg_mask * fg_bin):  # ignore if missing foreground
                logger.debug(f'Skipping label {label} due to empty fg')
                continue

            # find outer contour (parts):
            contours, _ = cv2.findContours(
                bg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # determine areas of parts:
            areas = [cv2.contourArea(contour) for contour in contours]
            total_area = sum(areas)
            if not total_area:  # ignore if too small
                continue
            
            # redraw label array
            contour_labels = np.zeros_like(bg_mask, np.uint8)
            for i, contour in enumerate(contours):
                cv2.drawContours(contour_labels, contours, i, i + 1, cv2.FILLED)
            order = range(len(contours))

            # convert to polygons
            for i in order:
                contour = contours[i]
                area = areas[i]
                if min_area and area < min_area and area / total_area < 0.1:
                    logger.debug(f'Label {label} contour {i} is too small ({area}/{total_area})')
                    continue
                
                # simplify shape: can produce invalid (self-intersecting) polygons:
                polygon = contour[:, 0, ::]  # already ordered x,y
                
                # simplify and validate:
                polygon = Polygon(polygon)
                if not polygon.is_valid:
                    logger.debug(explain_validity(polygon))
                polygon = self.__make_valid(polygon)
                if not polygon.is_valid:
                    logger.debug(explain_validity(polygon))
                poly = polygon.exterior.coords[:-1]  # keep open
                if len(poly) < 4:
                    logger.debug(f'Label {label} contour {i} has less than 4 points')
                    continue
                
                # get baseline segments intersecting with this line mask
                # and concatenate them from left to right
                if baselines is not None:
                    base = self.__join_baselines([baseline.intersection(polygon) for baseline in baselines if baseline.intersects(polygon)])
                    if base is not None:
                        base = base.coords
                else:
                    base = None
                results.append((label, poly, base))  # ty: ignore[invalid-argument-type]
                result_labels[contour_labels == i + 1] = len(results)
        return results, result_labels
        
    def __join_polygons(self, polygons: list[Polygon], scale: float = 20) -> Polygon:
        polygons = list(
            chain.from_iterable([
                (poly.geoms if poly.geom_type in ['MultiPolygon', 'GeometryCollection']else [poly])
                for poly in polygons
            ])
        )
        npoly = len(polygons)
        if npoly == 1:
            return polygons[0]
        
        # find min-dist path through all polygons (travelling salesman)
        pairs = combinations(range(npoly), 2)
        dists = np.eye(npoly, dtype=float)
        for i, j in pairs:
            dist = polygons[i].distance(polygons[j])
            dist = max(dist, 1e-5)  # if pair merely touches, we still need to get an edge
            dists[i, j] = dist
            dists[j, i] = dist
        dists = minimum_spanning_tree(dists, overwrite=True)
        
        # add bridge polygons (where necessary)
        max_dist = max(1.0, scale / 5)
        for prevp, nextp in zip(*dists.nonzero()):
            prevp = polygons[prevp]
            nextp = polygons[nextp]
            nearest = nearest_points(prevp, nextp)
            bridgep = LineString(nearest).buffer(max_dist, resolution=1)
            polygons.append(bridgep)
        jointp = unary_union(polygons)
        assert jointp.geom_type == 'Polygon', jointp.wkt
        if jointp.minimum_clearance < 1.0:
            # follow-up calculations will necessarily be integer;
            # so anticipate rounding here and then ensure validity
            jointp = Polygon(np.round(jointp.exterior.coords))
            jointp = self.__make_valid(jointp)
        return jointp
    
    def __clip_line(self, line_poly: np.ndarray, region_poly: np.ndarray) -> np.ndarray | None:
        def make_intersection(poly1, poly2):
            interp = poly1.intersection(poly2)
            # post-process
            if interp.is_empty or interp.area == 0.0:
                return None
            if interp.geom_type == 'GeometryCollection':
                # heterogeneous result: filter zero-area shapes (LineString, Point)
                interp = unary_union([geom for geom in interp.geoms if geom.area > 0])
            if interp.geom_type == 'MultiPolygon':
                # homogeneous result: construct convex hull to connect
                interp = self.__join_polygons(interp.geoms)
            if interp.minimum_clearance < 1.0:
                # follow-up calculations will necessarily be integer;
                # so anticipate rounding here and then ensure validity
                interp = Polygon(np.round(interp.exterior.coords))
                interp = self.__make_valid(interp)
            return interp

        childp = Polygon(line_poly)
        parentp = Polygon(region_poly)
        
        # ensure iNpUtilt coords have valid paths (without self-intersection)
        # (this can happen when shapes valid in floating point are rounded)
        childp = self.__make_valid(childp)
        parentp = self.__make_valid(parentp)
        if not childp.is_valid:
            return None
        if not parentp.is_valid:
            return None
        
        # check if clipping is necessary
        if childp.within(parentp):
            return childp.exterior.coords[:-1]
        
        # clip to parent
        interp = make_intersection(childp, parentp)
        if not interp:
            return None
        return interp.exterior.coords[:-1]  # keep open


@click.command('linesegmentation')
@click.argument('xml', type=click.Path(), callback=ClickUtils.glob, nargs=-1)
@click.option(
    '-i', '--image',
    help='Full suffix of the image files to be used. Defaults to the first image found with "<filename>.*png"',
    type=click.STRING
)
@click.option(
    '-o', '--output',
    help='Output directory for generated PAGE-XML files. If omitted, the input file will be overwritten.',
    type=click.Path(file_okay=False, path_type=Path)
)
@click.option(
    '-s', '--spread',
    help='Distance in points (pt) from the foreground to project text line (or text region) labels into the '
         'background for polygonal contours; If zero, project half a scale/capheight.',
    type=click.FLOAT,
    default=0.0,
    show_default=True,
)
@click.option(
    '-t', '--threads',
    help='Number of threads for concurrent region processing',
    type=click.INT,
    default=1,
    show_default=True
)
def cli(
    xml: list[Path],
    image: str | None,
    output: Path | None,
    spread: float, 
    threads: int,
    **kwargs
) -> None:
    """
    Compute baselines and polygons for existing TextRegions in PAGE-XML files.
    
    Only binary image inputs are supported!

    XML: One or more PAGE-XML paths. Use glob patterns in quotes to process multiple files.
    """
    if not xml:
        raise click.BadArgumentUsage('No input PAGE-XML files found')
    if output is not None:
        output.mkdir(exist_ok=True, parents=True)
        
    with Progress(
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn('[progress.description]{task.description}'),
    ) as progress:
        load_task = progress.add_task('Loading module', total=None)        
        pairs: list[tuple[Path, Path]] = []
        for xml_fp in xml:
            if image:
                img_fp = xml_fp.parent / (xml_fp.name.split('.')[0] + image)
            else:
                img_fp = next(xml_fp.parent.glob(f'{xml_fp.name.split('.')[0]}.*png'), None)
            if img_fp is None or not img_fp.exists():
                logger.error(f'No matching image found for PAGE-XML: {xml}')
                continue
            else:
                logger.debug(f'Found image {img_fp} for PAGE-XML {xml_fp}')
                pairs.append((xml_fp, img_fp))
        if not pairs:
            raise click.BadArgumentUsage('No files to process')
        
        segmenter = LineSegmentation(spread, threads)
        progress.remove_task(load_task)
        
        task = progress.add_task('Processing images', total=len(pairs))
        for xml_fp, img_fp in pairs:
            progress.update(task, description='/'.join(xml_fp.parts[-4:]))
            logger.info(f'Processing: {xml_fp} and {img_fp}')
            try:
                out_dir: Path = output or xml_fp.parent
                out_path: Path = out_dir / f'{xml_fp.name.split(".")[0]}.xml'
                segmenter.process(img_fp, xml_fp).save(out_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(f'Processing failed for {xml_fp}: {exc}')
            progress.advance(task)
        progress.update(task, description='Done')
