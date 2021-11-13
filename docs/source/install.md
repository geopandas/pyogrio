# Installation

## Requirements

Supports Python 3.6 - 3.9 and GDAL 2.4.x - 3.2.x
(prior versions will not be supported)

Reading to GeoDataFrames requires requires `geopandas>=0.8` with `pygeos` enabled.

## Installation

### Conda-forge

This package is available on [conda-forge](https://anaconda.org/conda-forge/pyogrio)
for Linux, MacOS, and Windows.

```bash
conda install -c conda-forge pyogrio
```

This requires compatible versions of `GDAL` and `numpy` from `conda-forge` for
raw I/O support and `geopandas`, `pygeos`, and their dependencies for GeoDataFrame
I/O support.

### PyPi

This package is not yet available on PyPI because it involves compiled binary
dependencies. We are planning to release this package on PyPI for Linux and MacOS.
We are unlikely to release Windows packages on PyPI in the near future due to
the complexity of packaging binary packages for Windows.

### Common installation errors

A driver error resulting from a `NULL` pointer exception like this:

```
pyogrio._err.NullPointerError: NULL pointer error

During handling of the above exception, another exception occurred:
...
pyogrio.errors.DriverError: Data source driver could not be created: GPKG
```

Is likely the result of a collision in underlying GDAL versions between `fiona`
(included in `geopandas`) and the GDAL version needed here. To get around it,
uninstall `fiona` then reinstall to use system GDAL:

```bash
pip uninstall fiona
pip install fiona --no-binary fiona
```

Then restart your interpreter.


## Development

Clone this repository to a local folder.

Install an appropriate distribution of GDAL for your system. `gdal-config` must
be on your system path.

Building `pyogrio` requires requires `Cython`, `numpy`, and `pandas`.

Run `python setup.py develop` to build the extensions in Cython.

Tests are run using `pytest`:

```bash
pytest pyogrio/tests
```

### Windows

Install GDAL from an appropriate provider of Windows binaries. We've heard that
the [OSGeo4W](https://trac.osgeo.org/osgeo4w/) works.

To build on Windows, you need to provide additional command-line parameters
because the location of the GDAL binaries and headers cannot be automatically
determined.

Assuming GDAL is installed to `c:\GDAL`, you can build as follows:

```bash
python -m pip install --install-option=build_ext --install-option="-IC:\GDAL\include" --install-option="-lgdal_i" --install-option="-LC:\GDAL\lib" --no-deps --force-reinstall --no-use-pep517 -e . -v
```

`GDAL_VERSION` environment variable must be if the version cannot be autodetected
using `gdalinfo.exe` (must be on your system `PATH` in order for this to work).

The location of the GDAL DLLs must be on your system `PATH`.

`--no-use-pep517` is required in order to pass additional options to the build
backend (see https://github.com/pypa/pip/issues/5771).

Also see `.github/test-windows.yml` for additional ideas if you run into problems.

Windows is minimally tested; we are currently unable to get automated tests
working on our Windows CI.