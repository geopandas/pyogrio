# CHANGELOG

## 0.4.0

### Major enhancements

-   index of GeoDataFrame created by `read_dataframe` can now optionally be set
    to the FID of the features that are read, as `int64` dtype. Note that some
    drivers start FID numbering at 0 whereas others start numbering at 1.
-   generalize check for VSI files from `/vsizip` to `/vsi` (#29)
-   add dtype for each field to `read_info` (#30)
-   support writing empty GeoDataFrames (#38)

### Breaking changes

-   `read` now also returns an optional FIDs ndarray in addition to meta,
    geometries, and fields; this is the 2nd item in the returned tuple.

### Potentially breaking changes

-   Consolided error handling to better use GDAL error messages and specific
    exception classes (#39). Note that this is a breaking change only if you are
    relying on specific error classes to be emitted.

### Bug fixes

-   use dtype `object` instead of `numpy.object` to eliminate deprecation warnings (#34)
-   raise error if layer cannot be opened (#35)

## 0.3.0

### Major enhancements

-   Auto-discovery of `GDAL_VERSION` on Windows, if `gdalinfo.exe` is discoverable
    on the `PATH`.
-   Addition of `read_bounds` function to read the bounds of each feature.
-   Addition of a `fids` keyword to `read` and `read_dataframe` to selectively
    read features based on a list of the FIDs.

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
