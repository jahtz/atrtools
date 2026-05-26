# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glob
import logging
from pathlib import Path

import click
from pypxml import PageXML, PageType
from rich.logging import RichHandler


logger: logging.Logger = logging.getLogger(__name__)


class clicku:
    """
    Collection of useful methods and callbacks for click interfaces
    """
    
    @staticmethod
    def callback_glob(
        ctx: click.Context, 
        param: click.Parameter, 
        patterns: list[str] | str
    ) -> list[Path]:
        if isinstance(patterns, str):
            patterns = [patterns]
        paths: list[Path] = []
        for pattern in patterns:
            if glob.has_magic(pattern):
                for match in glob.iglob(pattern, recursive=True):
                    p = Path(match)
                    if p.is_file():
                        paths.append(p.resolve())
            else:
                p = Path(pattern)
                if p.is_file() and p.exists():
                    paths.append(p.resolve())
        return paths
    
    @staticmethod
    def callback_pagetype(
        ctx: click.Context, 
        param: click.Parameter, 
        pagetypes: list[str] | str
    ) -> list[tuple[PageType, str | None]]:
        if isinstance(pagetypes, str):
            pagetypes = [pagetypes]
        result: list[tuple[PageType, str | None]] = []
        for pagetype in pagetypes:
            if '.' in pagetype:
                pt, t = pagetype.split('.')
                result.append((PageType[pt], t))
            else:
                result.append((PageType[pagetype], None))
        return result
    
    @staticmethod
    def setup_logging(level: int = 0) -> None:
        logging.basicConfig(
            level=max(10, 40 - (10 * level)),
            format='%(message)s', 
            datefmt='[%X]', 
            handlers=[RichHandler(markup=True, rich_tracebacks=True)]
        )
        logging.getLogger('pypxml').setLevel(max(30, 40 - (10 * level)))
        logger.info(f'Logging verbosity set to {logging.getLevelName(logger.getEffectiveLevel())}')


    @staticmethod
    def find_page_image_pairs(xml: Path, image_suffix: str | None = None) -> Path | None:
        """
        Finds corresponding image files for a PageXML file.
        Args:
            xml: PageXML file path.
            suffix: If provided, search for image with this suffix.
                    Else use filename provided in PageXMLs imageFilename attribute.
        Returns:
            The Path to the image file if found, else None.
        """
        if image_suffix:
            ifp = xml.parent / (xml.name.split('.')[0] + image_suffix)
            if ifp.exists():
                return ifp
        page = PageXML.open(xml)
        fn = page['imageFilename']
        if fn is None:
            return None
        ifp = xml.parent / fn
        return ifp if ifp.exists() else None
