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

Drivers that are not known to be supported are listed with `"?"` for capabilities.
Drivers that are known to support write capability end in `"w"`.

To find subsets of drivers that have known support:

```python
>>> list_drivers(read=True)
>>> list_drivers(write=True)
```

See the full list of [drivers](https://gdal.org/drivers/vector/index.html) for
more information about specific drivers, including their write support and
configuration options.

You can certainly try to read or write using unsupported drivers that are
available in your installation, but you may encounter errors.

Note: different drivers have different tolerance for mixed geometry types, e.g.,
MultiPolygon and Polygon in the same dataset. You will get exceptions or 
warnings if you attempt to write mixed geometries to a driver that does not 
support them.

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

# Outputs a dictionary with `crs`, `encoding`, `fields`, `geometry_type`, and `features`
{
  'crs': 'EPSG:4326',
  'encoding': 'UTF-8',
  'fields': array(['featurecla', 'scalerank', 'LABELRANK', ...], dtype=object),
  'dtypes': array(['int64', 'object', 'object', 'object', 'float64'], dtype=object),
  'geometry_type': 'Polygon',
  'features': 255
}
```

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

## Filter records by attribute value

You can use the `where` parameter to define a GDAL-compatible SQL WHERE query against
the records in the dataset:

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', where="POP_EST >= 10000000 AND POP_EST < 100000000")
```

See [GDAL docs](https://gdal.org/api/vector_c_api.html#_CPPv424OGR_L_SetAttributeFilter9OGRLayerHPKc)
for more information about restrictions of the `where` expression.

## Filter records by spatial extent

You can use the `bbox` parameter to select only those features that intersect
with the bbox.

```python
>>> read_dataframe('ne_10m_admin_0_countries.shp', bbox=(-140, 20, -100, 40))
```

Note: the `bbox` values must be in the same CRS as the dataset.

## Execute a sql query

You can use the `sql` parameter to execute a sql query on a dataset. 

Depending on the dataset, you can use different sql dialects. By default, if 
the dataset natively supports sql, the sql statement will be passed through 
as such. Hence, the sql query should be written in the relevant native sql
dialect (e.g. [GeoPackage](https://gdal.org/drivers/vector/gpkg.html)/
[Sqlite](https://gdal.org/drivers/vector/sqlite.html), 
[PostgreSQL](https://gdal.org/drivers/vector/pg.html)). If the datasource 
doesn't natively support sql (e.g. 
[ESRI Shapefile](https://gdal.org/drivers/vector/shapefile.html), 
[FlatGeobuf](https://gdal.org/drivers/vector/flatgeobuf.html)), you can choose 
between '[OGRSQL](https://gdal.org/user/ogr_sql_dialect.html#ogr-sql-dialect)' 
(the default) and  
'[SQLITE](https://gdal.org/user/sql_sqlite_dialect.html#sql-sqlite-dialect)'. 
For SELECT statements the 'SQLITE' dialect tends to provide more spatial 
features as all 
[spatialite](https://www.gaia-gis.it/gaia-sins/spatialite-sql-latest.html) 
functions can be used. 

You can combine a sql query with other parameters that will filter the 
dataset. When using ``columns``, ``skip_features``, ``max_features``, and/or 
``where`` it is important to note that they will be applied AFTER the sql 
statement, so these are some things you need to be aware of:
- if you specify an alias for a column in the sql statement, you need to 
  specify this alias when using the ``columns`` keyword.
- ``skip_features`` and ``max_features`` will be applied on the rows returned 
  by the sql query, not on the original dataset.

For the ``bbox`` parameter, depending on the combination of the dialect of the 
sql query and the dataset, a spatial index will be used or not, e.g.:
- ESRI Shapefile: spatial index is used with 'OGRSQL', not with 'SQLITE'.
- Geopackage: spatial index is always used.

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

## Write a GeoPandas GeoDataFrame

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

## Configuration options

It is possible to set
[GDAL configuration options](https://trac.osgeo.org/gdal/wiki/ConfigOptions) for
an entire session:

```python
>>> from pyogrio import set_gdal_config_options
>>> set_gdal_config_options({"CPL_DEBUG": True})
```

`True` / `False` values are automatically converted to `'ON'` / `'OFF'`.
