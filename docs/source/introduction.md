# Introduction to Pyogrio

## Display GDAL version

You can display the GDAL version that Pyogrio was compiled against by

```python
>>> pyogrio.__gdal_version__
```

## List available drivers

Use `pyogrio.list_drivers()` to list all available drivers in your installation
of GDAL. However, just because a driver is listed does not mean that it is
currently compatible with Pyogrio.

```{warning}
Not all geometry or field types may be supported for all drivers.
```

```python
>>> from pyogrio import list_drivers
>>> list_drivers()
{...'GeoJSON': 'rw', 'GeoJSONSeq': 'rw',...}
```

Drivers that support write capability in your version of GDAL end in `"w"`.
Certain drivers that are known to be unsupported in Pyogrio are disabled for
write capabilities.

NOTE: not all drivers support writing the contents of a GeoDataFrame; you may
encounter errors due to unsupported data types, unsupported geometry types,
or other driver-related errors when writing to a data source.

To find subsets of drivers that support read or write capabilities:

```python
>>> list_drivers(read=True)
>>> list_drivers(write=True)
```

See the full list of [drivers](https://gdal.org/drivers/vector/index.html) for
more information about specific drivers, including their write support and
configuration options.

The following drivers are known to be well-supported and tested in Pyogrio:

-   `ESRI Shapefile`
-   `FlatGeobuf`
-   `GeoJSON`
-   `GeoJSONSeq`
-   `GPKG`

## List available layers

To list layers available in a data source:

```python
>>> from pyogrio import list_layers
>>> list_layers('ne_10m_admin_0_countries.shp')

# Outputs ndarray with the layer name and geometry type for each layer
array([['ne_10m_admin_0_countries', 'Polygon']], dtype=object)
```

Some data sources (e.g., ESRI FGDB) support multiple layers, some of which may
be nonspatial. In this case, the geometry type will be `None`.

## Read basic information about a data layer

To list information about a data layer in a data source, use the name of the layer
or its index (0-based) within the data source. By default, this reads from the
first layer.

```python
>>> from pyogrio import read_info
>>> read_info('ne_10m_admin_0_countries.shp')

# Outputs a dictionary with `crs`, `driver`, `encoding`, `fields`, `geometry_type`, and
# `features`
{
  'crs': 'EPSG:4326',
  'encoding': 'UTF-8',
  'fields': array(['featurecla', 'scalerank', 'LABELRANK', ...], dtype=object),
  'dtypes': array(['int64', 'object', 'object', 'object', 'float64'], dtype=object),
  'geometry_type': 'Polygon',
  'features': 255,
  'driver': 'ESRI Shapefile',
}
```

NOTE: pyogrio will report `UTF-8` if either the native encoding is likely to be
`UTF-8` or GDAL can automatically convert from the detected native encoding to
`UTF-8`.

To read from a layer using name or index (the following are equivalent):

```python
>>>read_info('ne_10m_admin_0_countries.shp', layer='ne_10m_admin_0_countries')
>>>read_info('ne_10m_admin_0_countries.shp', layer=0)
```

## Read a data layer into a GeoPandas GeoDataFrame

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

## Read a subset of columns

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

## Read a subset of features

You can read a subset of features using `skip_features` and `max_features`.

To skip the first 10 features:

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', skip_features=10)
```

NOTE: Using this parameter may incur significant overhead if the driver does not
support the capability to randomly seek to a specific feature, because it will
need to iterate over all prior features.

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

NOTE: if `use_arrow` is `True`, `skip_features` and `max_features` will incur
additional overhead because all features up to the next batch size above
`max_features` (or size of data layer) will be read prior to slicing out the
requested range of features. If `max_features` is less than the maximum Arrow
batch size (65,536 features) only `max_features` will be read. All features
up to `skip_features` are read from the data source and later discarded because
the Arrow interface does not support randomly seeking a starting feature. This
overhead is in comparison to reading via Arrow without these parameters, which
is generally much faster than not using Arrow.

## Filter records by attribute value

You can use the `where` parameter to filter features in layer by attribute values. If
the data source natively supports SQL, its specific SQL dialect should be used
(eg. SQLite and GeoPackage:
[SQLITE](https://gdal.org/user/sql_sqlite_dialect.html#sql-sqlite-dialect), PostgreSQL).
If it doesn't, the [OGRSQL WHERE](https://gdal.org/user/ogr_sql_dialect.html#where)
syntax should be used. Note that it is not possible to overrule the SQL dialect, this is
only possible when you use the `sql` parameter.

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', where="POP_EST >= 10000000 AND POP_EST < 100000000")
```

## Filter records by spatial extent

You can use the `bbox` parameter to select only those features that intersect
with the bbox.

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', bbox=(-140, 20, -100, 40))
```

Note: the `bbox` values must be in the same CRS as the dataset.

Note: if GEOS is present and used by GDAL, only geometries that intersect `bbox`
will be returned; if GEOS is not available or not used by GDAL, all geometries
with bounding boxes that intersect this bbox will be returned.
`pyogrio.__gdal_geos_version__` will be `None` if GEOS is not detected.

## Filter records by a geometry

You can use the `mask` parameter to select only those features that intersect
with a Shapely (>= 2.0) geometry.

```python
>>> mask = shapely.Polygon(([-80,8], [-80, 10], [-85,10], [-85,8], [-80,8]))
>>> read_dataframe('ne_10m_admin_0_countries.shp', mask=mask)
```

Note: the `mask` values must be in the same CRS as the dataset.

If your mask geometry is in some other representation, such as GeoJSON, you will
need to convert it to a Shapely geometry before using `mask`.

```python
>>> mask_geojson = '{"type":"Polygon","coordinates":[[[-80.0,8.0],[-80.0,10.0],[-85.0,10.0],[-85.0,8.0],[-80.0,8.0]]]}'
>>> mask = shapely.from_geojson(mask_geojson)
>>> read_dataframe('ne_10m_admin_0_countries.shp', mask=mask)
```

Note: if GEOS is present and used by GDAL, only geometries that intersect `mask`
will be returned; if GEOS is not available or not used by GDAL, all geometries
with bounding boxes that intersect the bounding box of `mask` will be returned.
`pyogrio.__gdal_geos_version__` will be `None` if GEOS is not detected.

## Execute a sql query

You can use the `sql` parameter to execute a sql query on a dataset.

Depending on the dataset, you can use different sql dialects. By default, if
the dataset natively supports sql, the sql statement will be passed through
as such. Hence, the sql query should be written in the relevant native sql
dialect (e.g. [GeoPackage](https://gdal.org/drivers/vector/gpkg.html)/
[Sqlite](https://gdal.org/drivers/vector/sqlite.html),
[PostgreSQL](https://gdal.org/drivers/vector/pg.html)). If the data source
doesn't natively support sql (e.g.
[ESRI Shapefile](https://gdal.org/drivers/vector/shapefile.html),
[FlatGeobuf](https://gdal.org/drivers/vector/flatgeobuf.html)), you can choose
between '[OGRSQL](https://gdal.org/user/ogr_sql_dialect.html#ogr-sql-dialect)'
(the default) and
'[SQLITE](https://gdal.org/user/sql_sqlite_dialect.html#sql-sqlite-dialect)'.
For SELECT statements the 'SQLITE' dialect tends to provide more spatial
features as all
[spatialite](https://www.gaia-gis.it/gaia-sins/spatialite-sql-latest.html)
functions can be used. If gdal is not built with spatialite support in SQLite,
you can use `sql_dialect="INDIRECT_SQLITE"` to be able to use spatialite
functions on native SQLite files like Geopackage.

You can combine a sql query with other parameters that will filter the
dataset. When using `columns`, `skip_features`, `max_features`, and/or
`where` it is important to note that they will be applied AFTER the sql
statement, so these are some things you need to be aware of:

-   if you specify an alias for a column in the sql statement, you need to
    specify this alias when using the `columns` keyword.
-   `skip_features` and `max_features` will be applied on the rows returned
    by the sql query, not on the original dataset.

For the `bbox` parameter, depending on the combination of the dialect of the
sql query and the dataset, a spatial index will be used or not, e.g.:

-   ESRI Shapefile: spatial index is used with 'OGRSQL', not with 'SQLITE'.
-   Geopackage: spatial index is always used.

The following sql query returns the 5 Western European countries with the most
neighbours:

```python
>>> sql = """
        SELECT geometry, name,
               (SELECT count(*)
                  FROM ne_10m_admin_0_countries layer_sub
                 WHERE ST_Intersects(layer.geometry, layer_sub.geometry)) AS nb_neighbours
          FROM ne_10m_admin_0_countries layer
         WHERE subregion = 'Western Europe'
         ORDER BY nb_neighbours DESC
         LIMIT 5"""
>>> read_dataframe('ne_10m_admin_0_countries.shp', sql=sql, sql_dialect='SQLITE')
          NAME  nb_neighbours                            geometry
0       France             11  MULTIPOLYGON (((-54.11153 2.114...
1      Germany             10  MULTIPOLYGON (((13.81572 48.766...
2      Austria              9  POLYGON ((16.94504 48.60417, 16...
3  Switzerland              6  POLYGON ((10.45381 46.86443, 10...
4      Belgium              5  POLYGON ((2.52180 51.08754, 2.5...
```

## Force geometries to be read as 2D geometries

You can force a 3D dataset to 2D using `force_2d`:

```python
>>> df = read_dataframe('has_3d.shp')
>>> df.iloc[0].geometry.has_z
True

>>> df = read_dataframe('has_3d.shp', force_2d=True)
>>> df.iloc[0].geometry.has_z
False
```

## Read without geometry into a Pandas DataFrame

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

You can also read nonspatial tables, such as tables within an ESRI File Geodatabase
or a DBF file, directly into a Pandas `DataFrame`.

## Read feature bounds

You can read the bounds of all or a subset of features in the dataset in order
to create a spatial index of features without reading all underlying geometries.
This is typically 2-3x faster than reading full feature data, but the main
benefit is to avoid reading all feature data into memory for very large datasets.

```python
>>> from pyogrio import read_bounds
>>> fids, bounds = read_bounds('ne_10m_admin_0_countries.shp')
```

`fids` provide the global feature id of each feature.
`bounds` provide an ndarray of shape (4,n) with values for `xmin`, `ymin`,
`xmax`, `ymax`.

This function supports options to subset features from the dataset:

-   `skip_features`
-   `max_features`
-   `where`
-   `bbox`
-   `mask`

## Write a GeoPandas GeoDataFrame

You can write a `GeoDataFrame` `df` to a file as follows:

```python
>>> from pyogrio import write_dataframe
>>> write_dataframe(df, "/tmp/test.gpkg")
```

By default, the appropriate driver is inferred from the extension of the filename:

-   `.fgb`: [FlatGeobuf](https://gdal.org/drivers/vector/flatgeobuf.html)
-   `.geojson`: [GeoJSON](https://gdal.org/drivers/vector/geojson.html)
-   `.geojsonl`, `.geojsons`: [GeoJSONSeq](https://gdal.org/drivers/vector/geojsonseq.html)
-   `.gpkg`: [GPKG](https://gdal.org/drivers/vector/gpkg.html)
-   `.shp`: [ESRI Shapefile](https://gdal.org/drivers/vector/shapefile.html)

If you want to write another file format supported by GDAL or if you want to
overrule the default driver for an extension, you can specify the driver with the
`driver` keyword, e.g. `driver="GPKG"`.

## Appending to an existing data source

Certain drivers may support the ability to append records to an existing
data layer in an existing data source. See the
[GDAL driver listing](https://gdal.org/drivers/vector/index.html)
for details about the capabilities of a driver for your version of GDAL.

```
>>> write_dataframe(df, "/tmp/existing_file.gpkg", append=True)
```

NOTE: the data structure of the data frame you are appending to the existing
data source must exactly match the structure of the existing data source.

NOTE: not all drivers that support write capabilities support append
capabilities for a given GDAL version.

## Reading from compressed files / archives

GDAL supports reading directly from an archive, such as a zipped folder, without
the need to manually unpack the archive first. This is especially useful when
the dataset, such as a ESRI Shapefile, consists of multiple files and is
distributed as a zipped archive.

GDAL handles this through the concept of [virtual file systems](https://gdal.org/user/virtual_file_systems.html)
using a `/vsiPREFIX/..` path (for example `/vsizip/..`). For convenience,
pyogrio also supports passing the path with the more common URI syntax
using `zip://..`:

```python
>>> read_dataframe("/vsizip/ne_10m_admin_0_countries.zip")
>>> read_dataframe("zip://ne_10m_admin_0_countries.zip")
```

If your archive contains multiple datasets, you need to specify which one to use;
otherwise GDAL will default to the first one found.

```python
>>> read_dataframe("/vsizip/multiple_datasets.zip/a/b/test.shp")
>>> read_dataframe("zip://multiple_datasets.zip/a/b/test.shp")
>>> read_dataframe("zip://multiple_datasets.zip!a/b/test.shp")
```

Pyogrio will attempt to autodetect zip files if the filename or archive path
ends with `.zip` and will add the `/vsizip/` prefix for you, but you must use
`"!"` to denote the archive name in order to read a specific dataset within the
archive:

```python
>>> read_dataframe("multiple_datasets.zip!/a/b/test.shp")
```

## Reading from remote filesystems

GDAL supports several remote filesystems, such as S3, Google Cloud or Azure,
out of the box through the concept of virtual file systems. See
[GDAL's docs on network file systems](https://gdal.org/user/virtual_file_systems.html#network-based-file-systems)
for more details.
You can use GDAL's native `/vsi../` notation, but for convenience, pyogrio
also supports passing the path with the more common URI syntax:

```python
>>> read_dataframe("/vsis3/bucket/data.geojson")
>>> read_dataframe("s3://bucket/data.geojson")
```

It is also possible to combine multiple virtual filesystems, such as reading
a zipped folder (see section above) from a remote filesystem:

```python
>>> read_dataframe("vsizip/vsis3/bucket/shapefile.zip")
>>> read_dataframe("zip+s3://bucket/shapefile.zip")
```

You can also read from a URL with this syntax:

```python
>>> read_dataframe("https://s3.amazonaws.com/bucket/data.geojson")
>>> read_dataframe("zip+https://s3.amazonaws.com/bucket/shapefile.zip")
```

## Reading and writing DateTimes

GDAL only supports datetimes at a millisecond resolution. Reading data will thus
give at most millisecond resolution (`datetime64[ms]` data type). With pandas 2.0
`pyogrio.read_dataframe()` will return datetime data as `datetime64[ms]`
correspondingly. For previous versions of pandas, `datetime64[ns]` is used as
ms precision was not supported. When writing, only precision up to
ms is retained.

Not all file formats have dedicated support to store datetime data, like ESRI
Shapefile. For such formats, or if you require precision > ms, a workaround is to
convert the datetimes to string.

When you have datetime columns with time zone information, it is important to
note that GDAL only represents time zones as UTC offsets, whilst pandas uses
IANA time zones (via `pytz` or `zoneinfo`). As a result, even if a column in a
DataFrame contains datetimes in a single time zone, this will often still result
in mixed time zone offsets being written for time zones where daylight saving
time is used (e.g. +01:00 and +02:00 offsets for time zone Europe/Brussels).
When roundtripping through GDAL, the information about the original time zone
is lost, only the offsets can be preserved. By default,
{func}`pyogrio.read_dataframe()` will convert columns with mixed offsets to UTC
to return a datetime64 column. If you want to preserve the original offsets,
you can use `datetime_as_string=True` or `mixed_offsets_as_utc=False`.

## Dataset and layer creation options

It is possible to use dataset and layer creation options available for a given
driver in GDAL (see the relevant
[GDAL driver page](https://gdal.org/drivers/vector/index.html)). These
can be passed in as additional `kwargs` to `write_dataframe` or using
dictionaries for dataset or layer-level options.

Where possible, Pyogrio uses the metadata of the driver to determine if a
given option is for the dataset or layer level. For drivers where the same
option is available for both levels, you will need to use `dataset_options`
or `layer_options` to specify the correct level.

Option names are automatically converted to uppercase.

`True` / `False` values are automatically converted to `'ON'` / `'OFF'`.

For instance, you can use creation options to create a spatial index for a
[shapefile](https://gdal.org/drivers/vector/shapefile.html#layer-creation-options).

```python
>>> write_dataframe(df, "/tmp/test.shp", spatial_index=True)
```

You can use upper case option names and values to match the GDAL options exactly
(creation options are converted to uppercase by default):

```python
>>> write_dataframe(df, '/tmp/test.shp', SPATIAL_INDEX="YES")
```

You can also use a dictionary to specify either `dataset_options` or
`layer_options` as appropriate for the driver:

```python
>>> write_dataframe(df, '/tmp/test.shp', layer_options={"spatial_index": True})
```

```python
>>> write_dataframe(df, '/tmp/test.gpkg', dataset_options={"version": "1.0"}, layer_options={"geometry_name": "the_geom"})
```

## Reading from and writing to in-memory datasets

It is possible to read from a dataset stored as bytes:

```python
from io import BytesIO

# save a GeoJSON to bytes
geojson = """{
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": { },
            "geometry": { "type": "Point", "coordinates": [1, 1] }
        }
    ]
}"""

geojson_bytes = BytesIO(geojson.encode("UTF-8"))

df = read_dataframe(geojson_bytes)
```

Note: this may emit a `RuntimeWarning` where the in-memory dataset is detected
to be a particular format but lacks the expected file extension (e.g., `.gpkg`)
because the in-memory path automatically created by pyogrio does not include the
extension.

It is also possible to write a dataset to bytes, but driver must also be
specified, and layer name should be specified to avoid it being set to a random
character string:

```python
buffer = BytesIO()

write_dataframe(df, buffer, layer="my_layer", driver="GPKG")

out_bytes = buffer.getvalue()
```

Note: this is limited to single-file data formats (e.g., GPKG) and does not
support formats with multiple files (e.g., ESRI Shapefile).

It is also possible to use a `/vsimem/` in-memory dataset with other GDAL-based
packages that support the `/vsimem/` interface, such as the `gdal` package:

```python
from osgeo import gdal

write_dataframe(df, "/vsimem/test.gpkg", layer="my_layer", driver="GPKG")

# perform some operation using it
gdal.Rasterize("test.tif", "/vsimem/test.gpkg", outputType=gdal.GDT_Byte, noData=255, initValues=255, xRes=0.1, yRes=-0.1, allTouched=True, burnValues=1)

# release the memory using pyogrio
from pyogrio import vsi_unlink

vsi_unlink("/vsimem/test.gpkg")
```

Pyogrio can also read from a valid `/vsimem/` file created using a different
package.

It is the user's responsibility to clean up the in-memory filesystem; pyogrio
will not automatically release those resources.

## Configuration options

It is possible to set
[GDAL configuration options](https://trac.osgeo.org/gdal/wiki/ConfigOptions) for
an entire session:

```python
>>> from pyogrio import set_gdal_config_options
>>> set_gdal_config_options({"CPL_DEBUG": True})
```

`True` / `False` values are automatically converted to `'ON'` / `'OFF'`.
