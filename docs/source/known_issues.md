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

Date fields are not yet fully supported. These will be supported in a future
release.
