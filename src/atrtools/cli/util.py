# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from pypxml import PageXML


def find_image(xml: Path, suffix: str | None = None) -> Path | None:
    """
    Finds corresponding image files for a PageXML file.
    Args:
        xml: PageXML file path.
        suffix: If provided, search for image with this suffix.
                Else use filename provided in PageXMLs imageFilename attribute.
    Returns:
        The Path to the image file if found, else None.
    """
    if suffix:
        image = xml.parent / (xml.name.split('.')[0] + suffix)
        if image.exists():
            return image
    else:
        page = PageXML.open(xml)
        filename = page['imageFilename']
        if filename is None:
            return None
        image = xml.parent / filename
        if image.exists():
            return image
    return None
