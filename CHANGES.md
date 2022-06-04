# CHANGELOG

## 0.4.0

### Major enhancements

-   support for reading from file-like objects and in-memory buffers (#25)
-   index of GeoDataFrame created by `read_dataframe` can now optionally be set
    to the FID of the features that are read, as `int64` dtype. Note that some
    drivers start FID numbering at 0 whereas others start numbering at 1.
-   generalize check for VSI files from `/vsizip` to `/vsi` (#29)
-   add dtype for each field to `read_info` (#30)
-   support writing empty GeoDataFrames (#38)
-   support URI schemes (`zip://`, `s3://`) (#43)
-   add keyword to promote mixed singular/multi geometry column to multi geometry type (#56)
-   Python wheels built for Windows, MacOS (x86_64), and Linux (x86_64) (#49, #55, #57, #61, #63)
-   automatically prefix zip files with URI scheme (#68)
-   support use of a sql statement in read_dataframe (#70)
-   correctly write geometry type for layer when dataset has multiple geometry types (#82)
-   support reading `bool`, `int16`, `float32` into correct dtypes (#83)
-   add `layer_geometry_type` to `write_dataframe` to set geometry type for layer (#85)
-   Use certifi to set `GDAL_CURL_CA_BUNDLE` / `PROJ_CURL_CA_BUNDLE` defaults (#97)
-   automatically detect driver for `.geojson`, `.geojsonl` and `.geojsons` files (#101)

### Breaking changes

-   `read` now also returns an optional FIDs ndarray in addition to meta,
    geometries, and fields; this is the 2nd item in the returned tuple.

### Potentially breaking changes

-   Consolidated error handling to better use GDAL error messages and specific
    exception classes (#39). Note that this is a breaking change only if you are
    relying on specific error classes to be emitted.
-   by default, writing GeoDataFrames with mixed singular and multi geometry
    types will automatically promote to the multi type if the driver does not
    support mixed geometry types (e.g., `FGB`, though it can write mixed geometry
    types if `layer_geometry_type` is set to `"Unknown"`)
-   the geometry type of datasets with multiple geometry types will be set to
    `"Unknown"` unless overridden using `layer_geometry_type`. Note:
    `"Unknown"` may be ignored by some drivers (e.g., shapefile)

### Bug fixes

-   use dtype `object` instead of `numpy.object` to eliminate deprecation warnings (#34)
-   raise error if layer cannot be opened (#35)
-   fix passing gdal creation parameters in `write_dataframe` (#62)
-   fix passing kwargs to GDAL in `write_dataframe` (#67)

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
