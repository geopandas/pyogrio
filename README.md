# pyogrio - Vectorized spatial vector file format I/O using GDAL/OGR

This provides a
[GeoPandas](https://github.com/geopandas/geopandas)-oriented API to OGR vector
data sources, such as ESRI Shapefile, GeoPackage, and GeoJSON. This converts to
/ from `GeoPandas` `GeoDataFrame`s when the data source includes geometry and `Pandas`
`DataFrame`s otherwise.

WARNING: this is an early version and the API is subject to substantial change.

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
raw I/O support and `geopandas`, `pygeos` and their dependencies for GeoDataFrame
I/O support.

### PyPi

This package is not yet available on PyPi because it involves compiled binary
dependencies. We are planning to release this package on PyPi for Linux and MacOS.
We are unlikely to release Windows packages on PyPi in the near future due to
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

## Supported vector formats:

Full support:

-   [ESRI Shapefile](https://gdal.org/drivers/vector/shapefile.html)
-   [GeoPackage](https://gdal.org/drivers/vector/gpkg.html)
-   [GeoJSON](https://gdal.org/drivers/vector/geojson.html) / [GeoJSONSeq](https://gdal.org/drivers/vector/geojsonseq.html)

Read support:

-   [ESRI FileGDB (via OpenFileGDB)](https://gdal.org/drivers/vector/openfilegdb.html#vector-openfilegdb)
-   above formats using the [Virtual File System](https://gdal.org/user/virtual_file_systems.html#virtual-file-systems), which allows use of zipped data sources and directories

Other vector formats registered with your installation of GDAL should be supported for read access only; these have not been tested.

We may consider supporting write access to other widely used vector formats that have an available driver in GDAL. Please open an issue to suggest a format critical to your work.

We will most likely not consider supporting obscure, rarely-used, proprietary vector formats, especially if they require advanced GDAL installation procedures.

## Performance

Based on initial benchmarks using recent versions of `fiona`, `geopandas`, and `pygeos`:

Compared to `fiona`:

-   1.6x faster listing of layers in single-layer data source
-   1.6x - 5x faster reading of small data sources (Natural Earth 10m and 110m Admin 0 and Admin 1 levels)
-   9 - 14x faster writing of small data sources

Compared to `geopandas` in native `shapely` objects, converting data frame here to `pygeos` objects:

-   6.5 - 16.5x faster reading of data into geometry-backed data frames
-   15 - 26x faster writing of GeoDataFrames to shapefile / geopackage

## API

### Available drivers

Use `pyogrio.list_drivers()` to list all available drivers. However, just
because a driver is listed does not mean that it is currently compatible with
pyogrio. Not all field types or geometry types may be supported for all drivers.

```python
>>> from pyogrio import list_drivers
>>> list_drivers()
{...'GeoJSON': 'rw', 'GeoJSONSeq': 'rw',...}
```

Drivers that are not known to be supported are listed with `"?"` for capabilities.
Drivers that are known to support write capability end in `"w"`.

To find subsets of drivers that have known support:

```python
>>> list_drivers(read=True)
>>> list_drivers(write=True)
```

See full list of [drivers](https://gdal.org/drivers/vector/index.html) for more
information.

You can certainly try to read or write using unsupported drivers that are
available in your installation, but you may encounter errors.

Note: different drivers have different tolerance for mixed geometry types, e.g.,
MultiPolygon and Polygon in the same dataset. You will get exceptions if you
attempt to write mixed geometries to a driver that doesn't support them.

### Listing layers

To list layers available in a data source:

```python
>>> from pyogrio import list_layers
>>> list_layers('ne_10m_admin_0_countries.shp')

# Outputs ndarray with the layer name and geometry type for each layer
array([['ne_10m_admin_0_countries', 'Polygon']], dtype=object)
```

Some data sources (e.g., ESRI FGDB) support multiple layers, some of which may
be nonspatial. In this case, the geometry type will be `None`.

### Reading information about a data layer

To list information about a data layer in a data source, use the name of the layer
or its index (0-based) within the data source. By default, this reads from the
first layer.

```python
>>> from pyogrio import read_info
>>> read_info('ne_10m_admin_0_countries.shp')

# Outputs a dictionary with `crs`, `encoding`, `fields`, `geometry_type`, and `features`
{
  'crs': 'EPSG:4326',
  'encoding': 'UTF-8',
  'fields': array(['featurecla', 'scalerank', 'LABELRANK', ...], dtype=object),
  'geometry_type': 'Polygon',
  'features': 255
}
```

To read from a layer using name or index (the following are equivalent):

```python
>>>read_info('ne_10m_admin_0_countries.shp', layer='ne_10m_admin_0_countries')
>>>read_info('ne_10m_admin_0_countries.shp', layer=0)
```

### Reading into a GeoPandas GeoDataFrame

To read all features from a spatial data layer. By default, this operates on
the first layer unless `layer` is specified using layer name or index.

```python
>>> from pyogrio import read_dataframe
>>> read_dataframe('ne_10m_admin_0_countries.shp')

          featurecla  ...                                           geometry
0    Admin-0 country  ...  MULTIPOLYGON (((117.70361 4.16341, 117.70361 4...
1    Admin-0 country  ...  MULTIPOLYGON (((117.70361 4.16341, 117.69711 4...
2    Admin-0 country  ...  MULTIPOLYGON (((-69.51009 -17.50659, -69.50611...
3    Admin-0 country  ...  POLYGON ((-69.51009 -17.50659, -69.51009 -17.5...
4    Admin-0 country  ...  MULTIPOLYGON (((-69.51009 -17.50659, -69.63832...
..               ...  ...                                                ...
250  Admin-0 country  ...  MULTIPOLYGON (((113.55860 22.16303, 113.56943 ...
251  Admin-0 country  ...  POLYGON ((123.59702 -12.42832, 123.59775 -12.4...
252  Admin-0 country  ...  POLYGON ((-79.98929 15.79495, -79.98782 15.796...
253  Admin-0 country  ...  POLYGON ((-78.63707 15.86209, -78.64041 15.864...
254  Admin-0 country  ...  POLYGON ((117.75389 15.15437, 117.75569 15.151...
```

#### Subsets

You can read a subset of columns by including the `columns` parameter. This
only affects non-geometry columns:

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', columns=['ISO_A3'])
    ISO_A3                                           geometry
0      IDN  MULTIPOLYGON (((117.70361 4.16341, 117.70361 4...
1      MYS  MULTIPOLYGON (((117.70361 4.16341, 117.69711 4...
2      CHL  MULTIPOLYGON (((-69.51009 -17.50659, -69.50611...
3      BOL  POLYGON ((-69.51009 -17.50659, -69.51009 -17.5...
4      PER  MULTIPOLYGON (((-69.51009 -17.50659, -69.63832...
..     ...                                                ...
250    MAC  MULTIPOLYGON (((113.55860 22.16303, 113.56943 ...
251    -99  POLYGON ((123.59702 -12.42832, 123.59775 -12.4...
252    -99  POLYGON ((-79.98929 15.79495, -79.98782 15.796...
253    -99  POLYGON ((-78.63707 15.86209, -78.64041 15.864...
254    -99  POLYGON ((117.75389 15.15437, 117.75569 15.151...
```

You can read a subset of features using `skip_features` and `max_features`.

To skip the first 10 features:

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', skip_features=10)
```

NOTE: the index of the GeoDataFrame is based on the features that are read from
the file, it does not start at `skip_features`.

To read only the first 10 features:

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', max_features=10)
```

These can be combined to read defined ranges in the dataset, perhaps in multiple
processes:

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', skip_features=10, max_features=10)
```

### Filtering records by attribute value

You can use the `where` parameter to define a GDAL-compatible SQL WHERE query against
the records in the dataset:

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', where="POP_EST >= 10000000 AND POP_EST < 100000000")
```

See [GDAL docs](https://gdal.org/api/vector_c_api.html#_CPPv424OGR_L_SetAttributeFilter9OGRLayerHPKc)
for more information about restrictions of the `where` expression.

### Filtering records by spatial extent

You can use the `bbox` parameter to select only those features that intersect
with the bbox.

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', bbox=(-140, 20, -100, 40))
```

Note: the `bbox` values must be in the same CRS as the dataset.

### Ignoring geometry

You can omit the geometry from a spatial data layer by setting `read_geometry`
to `False`:

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', columns=['ISO_A3'], read_geometry=False)
    ISO_A3
0      IDN
1      MYS
2      CHL
3      BOL
4      PER
..     ...
250    MAC
251    -99
252    -99
253    -99
```

Any read operation which does not include a geometry column, either by reading
from a nonspatial data layer or by omitting the geometry column above, returns
a `Pandas` `DataFrame`.

### Forcing 2D

You can force a 3D dataset to 2D using `force_2d`:

```python
>>> df = read_dataframe('has_3d.shp')
>>> df.iloc[0].geometry.has_z
True

>>> df = read_dataframe('has_3d.shp', force_2d=True)
>>> df.iloc[0].geometry.has_z
False
```

#### Null values

Some data sources support NULL or otherwise unset field values. These cannot be properly
stored into the ndarray for certain types. If NULL or unset values are encountered,
the following occurs:

-   If the field is a string type, NULL values are represented as None
-   If the field is an integer type (np.int32, np.int64), the field data are
    re-cast to np.float64 values, and NULL values are represented as np.nan
-   If the field is a date or datetime type, the field is set as np.datetime64('NaT')

### Writing from a GeoPandas GeoDataFrame

To write a `GeoDataFrame` `df` to a file. `driver` defaults to `ESRI Shapefile`
(for now) but can be manually specified using one of the supported drivers for
writing (above):

```python
>>> from pyogrio import write_dataframe
>>> write_dataframe(df, '/tmp/test.shp', driver="GPKG")
```

The appropriate driver is also inferred automatically (where possible) from the
extension of the filename:
`.shp`: `ESRI Shapefile`
`.gpkg`: `GPKG`
`.json`: `GeoJSON`

### Reading feature bounds

You can read the bounds of all or a subset of features in the dataset in order
to create a spatial index of features without reading all underlying geometries.
This is typically 2-3x faster than reading full feature data, but the main
benefit is to avoid reading all feature data into memory for very large datasets.

```python
>>> from pyogrio import read_bounds
>>> fids, bounds = read_bounds('ne_10m_admin_0_countries.shp')
```

`fids` provide the global feature id of each feature.
`bounds` provide an ndarray of shape (4,n) with values for `xmin`, `ymin`, `xmax`, `ymax`.

This function supports options to subset features from the dataset:

-   `skip_features`
-   `max_features`
-   `where`
-   `bbox`

### Configuration options

It is possible to set [GDAL configuration options](https://trac.osgeo.org/gdal/wiki/ConfigOptions) for an entire session:

```python
>>> from pyogrio import set_gdal_config_options
>>> set_gdal_config_options({"CPL_DEBUG": True})
```

`True` / `False` values are automatically converted to `'ON'` / `'OFF'`.

### GDAL version

You can display the GDAL version that pyogrio was compiled against by

```python
>>> pyogrio.__gdal_version__
```

### Raw numpy-oriented I/O

see `pyogrio.raw` for numpy-oriented read / write interfaces to OGR data sources.

This may be useful for you if you want to work with the underlying arrays of
WKB geometries and field values outside of a `GeoDataFrame`.

NOTE: this may be migrated to an internal API in a future release.

## Limitations

### Measured geometries

Measured geometry types are not supported for reading or writing. These are not
supported by the GEOS library and cannot be converted to geometry objects in
GeoDataFrames.

These are automatically downgraded to their 2.5D (x,y, single z) equivalent and
a warning is raised.

To ignore this warning:

```python
>>> import warnings
>>> warnings.filterwarnings("ignore", message=".*Measured \(M\) geometry types are not supported.*")
```

### Curvilinear, triangle, TIN, and surface geometries

These geometry types are not currently supported. These are automatically
converted to their linear approximation when reading geometries from the data
layer.

## Known issues

`pyogrio` supports reading / writing data layers with a defined encoding. However,
DataFrames do not currently allow arbitrary metadata, which means that we are currently
unable to store encoding information for a data source. Text fields are read
into Python UTF-8 strings.

It does not currently validate attribute values or geometry types before attempting
to write to the output file. Invalid types may crash during writing with obscure
error messages.

Date fields are not currently supported properly. These will be supported in
a future release.

## How it works

`pyogrio` internally uses a numpy-oriented approach in Cython to read
information about data sources and records from spatial data layers. Geometries
are extracted from the data layer as Well-Known Binary (WKB) objects and fields
(attributes) are read into numpy arrays of the appropriate data type. These are
then converted to GeoPandas GeoDataFrames.

All records are read into memory, which may be problematic for very large data
sources. You can use `skip_features` / `max_features` to read smaller parts of
the file at a time.

The entire `GeoDataFrame` is written at once. Incremental writes or appends to
existing data sources are not supported.

## Comparison to Fiona

[`Fiona`](https://github.com/Toblerity/Fiona) is a full-featured Python library
for working with OGR vector data sources. It is **awesome**, has highly-dedicated
maintainers and contributors, and exposes more functionality than `pyogrio` ever will.
This project would not be possible without `Fiona` having come first.

`pyogrio` uses a vectorized (array-oriented) approach for reading and writing
spatial vector file formats, which enables faster I/O operations. It borrows
from the internal mechanics and lessons learned of `Fiona`. It uses a stateless
approach to reading or writing data; all data are read or written in a single
pass.

`Fiona` is a general purpose spatial format I/O library that is used within many
projects in the Python ecosystem. In contrast, `pyogrio` specifically targets
`GeoPandas` as an attempt to reduce the number of data transformations currently
required to read / write data between `GeoPandas` `GeoDataFrame`s and spatial
file formats using `Fiona` (the current default in GeoPandas).

## Credits

This project is made possible by the tremendous efforts of the GDAL, Fiona, and
Geopandas communities.

-   Core I/O methods and supporting functions adapted from [Fiona](https://github.com/Toblerity/Fiona)
-   Inspired by [Fiona PR](https://github.com/Toblerity/Fiona/pull/540/files)
