# Installation

## Requirements

Supports Python 3.8 - 3.10 and GDAL 2.4.x - 3.2.x
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

### PyPI

Ready-to-use (compiled) distributions are not yet available on PyPI because it
depends on including compiled binary dependencies. We are planning to release
compiled distributions on PyPI for Linux and MacOS relatively soon.

We are unlikely to release Windows packages on PyPI in the near future due to
the complexity of packaging binary packages for Windows. If you are interested
in helping us develop a packaging pipeline for Windows, please reach out!

### Common installation errors

If you install GeoPandas or Fiona using `pip`, you may encounter issues related
to incompatibility of the exact GDAL library pre-installed with Fiona and the
version of GDAL that gets compiled with Pyogrio (right now you do this manually).

This may show up as an exception like
this:

```
pyogrio.errors.DataSourceError: Could not obtain driver ...
```

To get around it, uninstall `fiona` then reinstall to use system GDAL:

```bash
pip uninstall fiona
pip install fiona --no-binary fiona
```

Then restart your interpreter. This ensures that both Pyogrio and Fiona use
exactly the same GDAL library.

## Development

Clone this repository to a local folder.

Install an appropriate distribution of GDAL for your system. Either
`gdal-config` must be on your system path (to automatically determine the
GDAL paths), or either the `GDAL_INCLUDE_PATH` and `GDAL_LIBRARY_PATH`
environment variables need to be set.

Building Pyogrio requires requires `Cython`, `numpy`, and `pandas`.

Run `python setup.py develop` to build the extensions in Cython.

Tests are run using `pytest`:

```bash
pytest pyogrio/tests
```

### Windows

Install GDAL from an appropriate provider of Windows binaries. We've heard that
the [OSGeo4W](https://trac.osgeo.org/osgeo4w/) works.

To build on Windows, you need to provide additional environment variables or
command-line parameters because the location of the GDAL binaries and headers
cannot be automatically determined.

Assuming GDAL is installed to `c:\GDAL`, you can set the `GDAL_INCLUDE_PATH`
and `GDAL_LIBRARY_PATH` environment variables and build as follows:

```bash
set GDAL_INCLUDE_PATH=C:\GDAL\include
set GDAL_LIBRARY_PATH=C:\GDAL\lib
python -m pip install --no-deps --force-reinstall --no-use-pep517 -e . -v
```

Alternatively, you can pass those options also as command-line parameters:

```bash
python -m pip install --install-option=build_ext --install-option="-IC:\GDAL\include" --install-option="-lgdal_i" --install-option="-LC:\GDAL\lib" --no-deps --force-reinstall --no-use-pep517 -e . -v
```

The location of the GDAL DLLs must be on your system `PATH`.

`--no-use-pep517` is required in order to pass additional options to the build
backend (see https://github.com/pypa/pip/issues/5771).

Also see `.github/test-windows.yml` for additional ideas if you run into problems.

Windows is minimally tested; we are currently unable to get automated tests
working on our Windows CI.

## GDAL and PROJ data files

GDAL requires certain files to be present within a GDAL data folder, as well
as a PROJ data folder. These folders are normally detected automatically.

If you have an unusual installation of GDAL and PROJ, you may need to set
additional environment variables at **runtime** in order for these to be
correctly detected by GDAL:

-   set `GDAL_DATA` to the folder containing the GDAL data files (e.g., contains `header.dxf`)
    within the installation of GDAL that is used by Pyogrio.
-   set `PROJ_LIB` to the folder containing the PROJ data files (e.g., contains `proj.db`)
