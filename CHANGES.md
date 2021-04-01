# CHANGELOG

## 0.2.0

### Major enhancements

-   initial support for building on Windows.
-   Windows: enabled search for GDAL dll directory for Python >= 3.8.
-   Addition of `where` parameter to `read` and `read_dataframe` to enable GDAL-compatible
    SQL WHERE queries to filter data sources.
-   Addition of `force_2d` parameter to `read` and `read_dataframe` to force
    coordinates to always be returned as 2 dimensional, dropping the 3rd dimension
    if present.
-   Addition of `bbox` parameter to `read` and `read_dataframe` to select only
    the features in the dataset that intersect the bbox.
-   Addition of `set_gdal_config_options` to set GDAL configuration options and
    `get_gdal_config_option` to get a GDAL configuration option.
-   Addition of `pyogrio.__gdal_version__` attribute to return GDAL version tuple
    and `__gdal_version_string__` to return string version.
-   Addition of `list_drivers` function to list all available GDAL drivers.
-   Addition of read and write support for `FlatGeobuf` driver when available in GDAL.

## 0.1.0

### Major enhancements

-   Addition of `list_layers` to list layers in a data source.
-   Addition of `read_info` to read basic information for a layer.
-   Addition of `read_dataframe` to read from supported file formats (Shapefile, GeoPackage, GeoJSON) into GeoDataFrames.
-   Addition of `write_dataframe` to write GeoDataFrames into supported file formats.
