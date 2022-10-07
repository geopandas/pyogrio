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

These are automatically downgraded to their 2.5D (x,y, single z) equivalent and
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
give at most millisecond resolution (`datetime64[ms]` data type), even though
the data is cast `datetime64[ns]` data type when reading into a data frame
using `pyogrio.read_dataframe()`. When writing, only precision up to ms is retained.

Not all file formats have dedicated support to store datetime data, like ESRI 
Shapefile. For such formats, or if you require precision > ms, a workaround is to
convert the datetimes to string.

Timezone information is ignored at the moment, both when reading and when writing
datetime columns.
