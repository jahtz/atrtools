# ATRtools

[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Collection of useful tools for ATR and PAGE-XML


## Setup

```bash
uv tool install git+https://github.com/jahtz/atrtools
```


## Usage

```
atrtools [OPTIONS] COMMAND [ARGS]...
```

```txt
$ atrtools --help
Usage: atrtools [OPTIONS] COMMAND [ARGS]...

  Collection of useful tools for ATR and PAGE-XML

Options:
  -v, --verbose  Set the verbosity level. Default: ERROR. Use -v for WARNING,
                 -vv for INFO, -vvv for DEBUG.

Commands:
  linesegmentation  Compute baselines and polygons.
  regionshrink      Shrink existing regions.
```


### Line Segmentation

Based on [CIS OCR-D](https://github.com/bertsky/ocrd_cis/):

```txt
$ atrtools linesegmentation --help
Usage: atrtools linesegmentation [OPTIONS] [XML]...

  Compute baselines and polygons for existing TextRegions in PAGE-XML files.

  Only binary image inputs are supported!

  XML: One or more PAGE-XML paths. Use glob patterns in quotes to process
  multiple files.

Options:
  -i, --image TEXT        Full suffix of the image files to be used. Defaults
                          to the first image found with "<filename>.*png"
  -o, --output DIRECTORY  Output directory for generated PAGE-XML files. If
                          omitted, the input file will be overwritten.
  -s, --spread FLOAT      Distance in points (pt) from the foreground to
                          project text line (or text region) labels into the
                          background for polygonal contours; If zero, project
                          half a scale/capheight.  [default: 0.0]
  -t, --threads INTEGER   Number of threads for concurrent region processing
                          [default: 1]
```


### Region Shrinking

```txt
$ atrtools regionshrink --help
Usage: atrtools regionshrink [OPTIONS] [XML]...

  Shrink existing regions to their contents in PAGE-XML files.

  Only binary image inputs are supported!

  XML: One or more PAGE-XML paths. Use glob patterns in quotes to process
  multiple files.

Options:
  -i, --image TEXT             Full suffix of the image files to be used.
                               Defaults to the first image found with
                               "<filename>.*png"
  -o, --output DIRECTORY       Output directory for generated PAGE-XML files.
                               If omitted, the input file will be overwritten.
  -p, --padding INTEGER        Padding between region borders and its content
                               in pixels.  [default: 5]
  -s, --smoothing FLOAT        Smoothing, calculated as the factor of the
                               average glyph size. Prevents regions cutting
                               between text.  [default: 1.0]
  -m, --mode [merge|largest]   Shrinking mode to use for regions. "merge"
                               merges all resulting polygons of each region
                               after shrinking. "largest" keeps only the
                               largest resulting polygon of each region after
                               shrinking.  [default: merge]
  -b, --bbox PageType          Draw a minimal bounding box for a specific
                               region after shrinking. Should be of format
                               "PageType" or "PageType.subtype". Multiple
                               regions can be specified. Examples: "-b
                               ImageRegion", "-b TextRegion.paragraph"
  -e, --exclude PageType       Exclude a specific region from shrinking.
                               Should be of format "PageType" or
                               "PageType.subtype". Multiple excludes can be
                               specified. Examples: "-e ImageRegion", "-e
                               TextRegion.paragraph"
  -t, --threads INTEGER RANGE  Number of threads for concurrent region
                               processing  [default: 1; x>=1]
```


## ZPD

Developed at Centre for [Philology and Digitality](https://www.uni-wuerzburg.de/en/zpd/) (ZPD), [University of Würzburg](https://www.uni-wuerzburg.de/en/).
