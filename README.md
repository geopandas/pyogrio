# pyogrio - Vectorized vector I/O using GDAL

Supports Python 3.8 and GDAL 2.4.x

## Goals

### Short term

Stub out enough of a minimal API for reading and writing from GDAL OGR to
establish a rough baseline of how fast vectorized operations like this can be
when built with Cython / Python. This will be used as a comparison of I/O
performance with Fiona to see if the approach can / should be ported there.

### Long term

Provide faster I/O using GDAL OGR to read / write vector geospatial files
using vectorized geometries and attributes.

Intended for use with libraries that consume WKB for their internal constructs,
such as [pygeos](https://github.com/pygeos/pygeos).

Geometries will be read from sources into ndarray of WKB.
Attributes will be read into ndarrays.

## Assumptions

Attributes have consistent type across all records.

## WORK IN PROGRESS

This is building toward a proof of concept, it is not meant for production use. Depending on
how this approach works out, it may be refactored in to the core of Fiona.

Right now, this borrows heavily from implementations in Fiona.

## Credits

-   Adapted from [fiona](https://github.com/Toblerity/Fiona)
-   Inspired by [fiona PR](https://github.com/Toblerity/Fiona/pull/540/files)
