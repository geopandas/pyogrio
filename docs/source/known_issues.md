# Limitations and Known Issues

## Support for null values

Some data sources support NULL or otherwise unset field values. These cannot be
properly stored into the ndarray for certain types. If NULL or unset values are
encountered, the following occurs:

-   If the field is a string type, NULL values are represented as None
-   If the field is a boolean or an integer type (np.int32, np.int64), the field
    data are re-cast to np.float64 values, and NULL values are represented as
    np.nan
-   If the field is a date or datetime type, the field is set as np.datetime64('NaT')

Note: detection of NULL or otherwise unset field values is limited to the subset
of records that are read from the data layer, which means that reading different
subsets of records may yield different data types for the same columns. You
can use `read_info()` to determine the original data types of each column.

## No support for measured geometries

Measured geometry types are not supported for reading or writing. These are not
supported by the GEOS library and cannot be converted to geometry objects in
GeoDataFrames.

These are automatically downgraded to their 3D (x,y,z) equivalent and
a warning is raised.

To ignore this warning:

```python
>>> import warnings
>>> warnings.filterwarnings("ignore", message=".*Measured \(M\) geometry types are not supported.*")
```

## No support for curvilinear, triangle, TIN, and surface geometries

Pyogrio does not support curvilinear, triangle, TIN, and surface geometries.
These are automatically converted to their linear approximation when reading
geometries from the data layer.

## Character encoding

Pyogrio supports reading / writing data layers with a defined encoding. However,
DataFrames do not currently allow arbitrary metadata, which means that we are
currently unable to store encoding information for a data source. Text fields
are read into Python UTF-8 strings.

## No validation of geometry or field types

Pyogrio does not currently validate attribute values or geometry types before
attempting to write to the output file. Invalid types may crash during writing
with obscure error messages.

## Support for reading and writing DateTimes

GDAL only supports datetimes at a millisecond resolution. Reading data will thus
give at most millisecond resolution (`datetime64[ms]` data type). With pandas 2.0
`pyogrio.read_dataframe()` will return datetime data as `datetime64[ms]` 
correspondingly. For previous versions of pandas, `datetime64[ns]` is used as 
ms precision was not supported. When writing, only precision up to 
ms is retained.

Not all file formats have dedicated support to store datetime data, like ESRI
Shapefile. For such formats, or if you require precision > ms, a workaround is to
convert the datetimes to string.

Timezone information is preserved where possible, however GDAL only represents
time zones as UTC offsets, whilst pandas uses IANA time zones (via `pytz` or 
`zoneinfo`). This means that dataframes with columns containing multiple offsets 
e.g. when switching from standard time to summer time will be written correctly,
but when read via `pyogrio.read_dataframe()` will be returned in UTC time, as
there is no way to reconstruct the original timezone from the individual offsets
present.

## Support for OpenStreetMap (OSM) data

OpenStreetMap data do not natively support calculating the feature count by data
layer due to the internal data structures. To get around this, Pyogrio iterates
over all features first to calculate the feature count that is used to allocate
arrays that contain the geometries and attributes read from the data layer, and
then iterates over all feature again to populate those arrays. Further, data
within the file are not structured at the top level to support fast reading by
layer, which means that reading data by layer may need to read all records
within the data source, not just those belonging to a particular layer. This is
inefficient and slow, and is exacerbated when attemping to read from
remotely-hosted data sources rather than local files.

You may also be instructed by GDAL to enable interleaved reading mode via an
error message when you try to read a large file without it, which you can do in
one of two ways:

1. Set config option used for all operations

```python
from pyogrio import set_gdal_config_options

set_gdal_config_options({"OGR_INTERLEAVED_READING": True})
```

2. Set dataset open option

```python

from pyogrio import read_dataframe

df = read_dataframe(path, INTERLEAVED_READING=True)
```

We recommend the following to sidestep performance issues:

-   download remote OSM data sources to local files before attempting
    to read
-   the `use_arrow=True` option may speed up reading from OSM files
-   if possible, use a different tool such as `ogr2ogr` to translate the OSM
    data source into a more performant format for reading by layer, such as GPKG

## Incorrect results when using a spatial filter and Arrow interface

Due to [a bug in GDAL](https://github.com/OSGeo/gdal/issues/8347), when using
the Arrow interface (e.g., via `use_arrow` on `read_dataframe`) certain drivers
(e.g., GPKG, FlatGeobuf, Arrow, Parquet) returned features whose bounding boxes
intersected the bounding box specified by `bbox` or `mask` geometry instead of
those whose geometry intersected the `bbox` or `mask`.

A fix is expected in GDAL 3.8.0.
