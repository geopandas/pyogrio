# pyogrio - Vectorized spatial vector file format I/O using GDAL/OGR

This provides an _experimental_
[GeoPandas](https://github.com/geopandas/geopandas)-oriented API to OGR vector
data sources, such as ESRI Shapefile, GeoPackage, and GeoJSON. This converts to
/ from `GeoPandas` `GeoDataFrame`s when the data source includes geometry and `Pandas`
`DataFrame`s otherwise.

WARNING: this is an early version and the API is subject to substantial change.

### Comparison to Fiona

[`Fiona`](https://github.com/Toblerity/Fiona) is a full-featured Python library
for working with OGR vector data sources. It is **awesome**, has highly-dedicated
maintainers and contributors, and exposes more functionality than `pyogrio` ever will.
This project would not be possible without `Fiona` having come first.

`pyogrio` is an experimental approach that uses a vectorized (array-oriented)
approach for reading and writing spatial vector file formats, which enables faster
I/O operations. It borrows from the internal mechanics and lessons learned of
`Fiona`.

`Fiona` is a general purpose spatial format I/O library that is used within many
projects in the Python ecosystem. In contrast, `pyogrio` specifically targets
`GeoPandas` as an attempt to reduce the number of data transformations currently
required to read / write data between `GeoPandas` `GeoDataFrame`s and spatial
file formats using `Fiona` (the current default in GeoPandas).

## Requirements

Supports Python 3.8 and GDAL 2.4.x
(versions of GDAL > 2.4.x may work, prior versions will not be supported)

Requires GeoPandas >= 0.8 with `pygeos` enabled.

We plan to support newer versions of GDAL in future releases.

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

### Listing layers

To list layers available in a data source:

```python
> from pyogrio import list_layers
> list_layers('ne_10m_admin_0_countries.shp')

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
> from pyogrio import read_info
> read_info('ne_10m_admin_0_countries.shp')

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
read_info('ne_10m_admin_0_countries.shp', layer='ne_10m_admin_0_countries')
read_info('ne_10m_admin_0_countries.shp', layer=0)
```

### Reading into a GeoPandas GeoDataFrame

To read all features from a spatial data layer. By default, this operates on
the first layer unless `layer` is specified using layer name or index.

```python
> from pyogrio import read_dataframe
> read_dataframe('ne_10m_admin_0_countries.shp')

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

You can read a subset of columns by including the `columns` parameter. This
only affects non-geometry columns:

```python
> read_dataframe('ne_10m_admin_0_countries.shp', columns=['ISO_A3'])
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

You can omit the geometry from a spatial data layer by setting `read_geometry`
to `False`:

```python
> read_dataframe('ne_10m_admin_0_countries.shp', columns=['ISO_A3'], read_geometry=False)
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

You can force a 3D dataset to 2D using `force_2d`:

```python
> df = read_dataframe('has_3d.shp')
> df.iloc[0].geometry.has_z
True

> df = read_dataframe('has_3d.shp', force_2d=True)
> df.iloc[0].geometry.has_z
False
```

Any read operation which does not include a geometry column, either by reading
from a nonspatial data layer or by omitting the geometry column above, returns
a `Pandas` `DataFrame`.

You can read a subset of features using `skip_features` and `max_features`.

To skip the first 10 features:

```python
read_dataframe('ne_10m_admin_0_countries.shp', skip_features=10)
```

NOTE: the index of the GeoDataFrame is based on the features that are read from
the file, it does not start at `skip_features`.

To read only the first 10 features:

```python
read_dataframe('ne_10m_admin_0_countries.shp', max_features=10)
```

These can be combined to read defined ranges in the dataset, perhaps in multiple
processes:

```python
read_dataframe('ne_10m_admin_0_countries.shp', skip_features=10, max_features=10)
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
> from pyogrio import write_dataframe
> write_dataframe(df, '/tmp/test.shp', driver="GPKG")
```

### Raw numpy-oriented I/O

see `pyogrio.raw` for numpy-oriented read / write interfaces to OGR data sources.

This may be useful for you if you want to work with the underlying arrays of
WKB geometries and field values outside of a `GeoDataFrame`.

NOTE: this may be migrated to an internal API in a future release.

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

The entire `GeoDataFrame` is written at once. Incremental writes or appends to
existing data sources are not supported.

## How it works

`pyogrio` internally uses a numpy-oriented approach in Cython to read
information about data sources and records from spatial data layers. Geometries
are extracted from the data layer as Well-Known Binary (WKB) objects and fields
(attributes) are read into numpy arrays of the appropriate data type. These are
then converted to GeoPandas GeoDataFrames.

All records are read into memory, which may be problematic for very large data
sources. You can use `skip_features` / `max_features` to read smaller parts of
the file at a time.

## Installation / development

Clone this repository to a local folder.

Right now, this requires system GDAL 2.4. See `install_extras` in the `setup.py`
for additional dependencies.

Run `python setup.py develop` to build the extensions in Cython.

Test datasets are downloaded and placed into `tests/fixtures/datasets` (each gets its own folder):

[Natural Earth](https://www.naturalearthdata.com/downloads/):

-   [Admin 0 (countries) at 1:110m](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip)
-   [Admin 0 (countries at 1:10m)](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip)
-   [Admin 1 (states / provinces) at 1:110m](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_1_states_provinces.zip)
-   [Admin 1 (states / provinces) at 1:10m](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_1_states_provinces.zip)

Hydrography:

-   [Watershed boundaries](https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GDB/WBD_17_HU2_GDB.zip)
-   [Flowlines, waterbodies, etc](https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_1704_HU4_GDB.zip)

## Common errors

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

```
pip uninstall fiona
pip install fiona --no-binary fiona
```

Then restart your interpreter.

## Credits

-   Adapted from [fiona](https://github.com/Toblerity/Fiona)
-   Inspired by [fiona PR](https://github.com/Toblerity/Fiona/pull/540/files)

Right now, this borrows heavily from implementations in Fiona.
