# Installation

## Requirements

Supports Python 3.10 - 3.14 and GDAL 3.6.x - 3.11.x

Reading to GeoDataFrames requires `geopandas>=0.12` with `shapely>=2`.

Additionally, installing `pyarrow` in combination with GDAL 3.6+ enables
a further speed-up when specifying `use_arrow=True`.

## Installation

### Conda-forge

This package is available on [conda-forge](https://anaconda.org/conda-forge/pyogrio)
for Linux, MacOS, and Windows.

```bash
conda install -c conda-forge pyogrio
```

This requires compatible versions of `GDAL` and `numpy` from `conda-forge` for
raw I/O support and `geopandas` and their dependencies for GeoDataFrame
I/O support. By default, the `GDAL` package on conda-forge already supports a
wide range of vector formats. If needed, you can install additional drivers by
installing the associated
[conda-forge package](https://gdal.org/en/latest/download.html#conda). The
following packages are currently available to install extra vector drivers:

-   `libgdal-arrow-parquet` ((Geo)Parquet and (Geo)Arrow IPC)
-   `libgdal-pg` (PostgreSQL / PostGIS)
-   `libgdal-xls` (XLS - MS Excel format)

### PyPI

This package is available on [PyPI](https://pypi.org/project/pyogrio/) for Linux,
MacOS, and Windows.

```bash
pip install pyogrio
```

This installs binary wheels that include GDAL.

If you get installation errors about Cython or GDAL not being available, this is
most likely due to the installation process falling back to installing from the
source distribution because the available wheels are not compatible with your
platform.

The binary wheels available on PyPI include the core GDAL drivers (GeoJSON,
ESRI Shapefile, GPKG, FGB, OpenFileGDB, etc) but do not include more advanced
drivers such as LIBKML and Spatialite. If you need such drivers, we recommend
that you use conda-forge to install pyogrio as explained above.

### Troubleshooting installation errors

If you install GeoPandas or Fiona using `pip`, you may encounter issues related
to incompatibility of the exact GDAL library pre-installed with Fiona and the
version of GDAL that gets compiled with Pyogrio.

This may show up as an exception like this for a supported driver (e.g.,
`ESRI Shapefile`):

```Python
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

Install an appropriate distribution of GDAL for your system. Either `gdal-config` must
be on your system path (to automatically determine the GDAL paths), or either the
`GDAL_INCLUDE_PATH`, `GDAL_LIBRARY_PATH`, and `GDAL_VERSION` environment variables need
to be set. Specific instructions on how to install these dependencies on Windows can be
found below.

Building Pyogrio requires requires `Cython`, `numpy`, and `pandas`.

Pyogrio follows the [GeoPandas Style Guide](https://geopandas.org/en/stable/community/contributing.html#style-guide-linting)
and uses `Ruff` to ensure consistent formatting.

It is recommended to install `pre-commit` and register its hooks so the formatting is
automatically verified when you commit code.

```
pre-commit install
```

Run `python setup.py develop` to build the extensions in Cython.

Tests are run using `pytest`:

```bash
pytest pyogrio/tests
```

### Windows

There are different ways to install the necessary dependencies and setup your local
development environment on windows.

#### vcpkg

[vcpkg](https://vcpkg.io/en/index.html) is used to build pyogrio from source
as part of creating the Pyogrio Python wheels for Windows. You can install
GDAL and other dependencies using vcpkg, and then build Pyogrio from source.

See `.github/workflows/release.yml` for details about how vcpkg is used as part
of the wheel-building process.

We do not yet have instructions on building Pyogrio from source using vcpkg for
local development; please feel free to contribute additional documentation!

#### OSGeo4W

You can also install GDAL from an appropriate provider of Windows binaries. We've heard
that the [OSGeo4W](https://trac.osgeo.org/osgeo4w/) works.

To build on Windows, you need to provide additional environment variables or
command-line parameters because the location of the GDAL binaries and headers
cannot be automatically determined.

Assuming GDAL 3.8.3 is installed to `c:\GDAL`, you can set the `GDAL_INCLUDE_PATH`,
`GDAL_LIBRARY_PATH` and `GDAL_VERSION` environment variables and build as follows:

```bash
set GDAL_INCLUDE_PATH=C:\GDAL\include
set GDAL_LIBRARY_PATH=C:\GDAL\lib
set GDAL_VERSION=3.8.3
python -m pip install --no-deps --force-reinstall --no-use-pep517 -e . -v
```

Alternatively, you can pass those options also as command-line parameters:

```bash
python -m pip install --install-option=build_ext --install-option="-IC:\GDAL\include" --install-option="-lgdal_i" --install-option="-LC:\GDAL\lib" --install-option="--gdalversion=3.8.3" --no-deps --force-reinstall --no-use-pep517 -e . -v
```

The location of the GDAL DLLs must be on your system `PATH`.

`--no-use-pep517` is required in order to pass additional options to the build
backend (see https://github.com/pypa/pip/issues/5771).

#### Conda

It is also possible to install the necessary dependencies using conda.

After cloning the environment, you can create a conda environment with the necessary
dependencies like this:

```
conda env create -f environment-dev.yml
```

Before being able to build on Windows, you need to set some additional environment
variables because the location of the GDAL binaries and headers cannot be
automatically determined.

After activating the `pyogrio-dev` environment the `CONDA_PREFIX` environment variable
will be available. Assuming GDAL 3.8.3 is installed, you will be able to set the
necessary environment variables as follows:

```bash
set GDAL_INCLUDE_PATH=%CONDA_PREFIX%\Library\include
set GDAL_LIBRARY_PATH=%CONDA_PREFIX%\Library\lib
set GDAL_VERSION=3.8.3
```

Now you should be able to run `python setup.py develop` to build the extensions in
Cython.

## GDAL and PROJ data files

GDAL requires certain files to be present within a GDAL data folder, as well
as a PROJ data folder. These folders are normally detected automatically.

If you have an unusual installation of GDAL and PROJ, you may need to set
additional environment variables at **runtime** in order for these to be
correctly detected by GDAL:

-   set `GDAL_DATA` to the folder containing the GDAL data files (e.g., contains `header.dxf`)
    within the installation of GDAL that is used by Pyogrio.
-   set `PROJ_LIB` to the folder containing the PROJ data files (e.g., contains `proj.db`)
