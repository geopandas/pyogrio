# CHANGELOG

## 0.5.0 (2023-01-??)

### Major enhancements

-   Support for reading based on Arrow as the transfer mechanism of the data
    from GDAL to Python (requires GDAL >= 3.6 and `pyarrow` to be installed).
    This can be enabled by passing `use_arrow=True` to `pyogrio.read_dataframe`
    (or by using `pyogrio.raw.read_arrow` directly), and provides a further
    speed-up (#155, #191).
-   Support for appending to an existing data source when supported by GDAL by
    passing `append=True` to `pyogrio.write_dataframe` (#197).

### Improvements

-   It is now possible to pass GDAL's dataset creation options in addition
    to layer creation options in `pyogrio.write_dataframe` (#189).
-   When specifying a subset of `columns` to read, unnecessary IO or parsing
    is now avoided (#195).
-   In floating point columns, NaN values are now by default written as "null"
    instead of NaN, but with an option to control this (pass `nan_as_null=False`
    to keep the previous behaviour) (#190).

### Packaging

-   The GDAL library included in the wheels is updated from 3.4 to GDAL 3.6.2,
    and is now built with GEOS and with sqlite with rtree support enabled
    (e.g. allowing to write a spatial index for GeoPackage).
-   Wheels are now available for Python 3.11.
-   Wheels are now available for MacOS arm64.

## 0.4.2 (2022-10-06)

### Improvements

-   new `get_gdal_data_path()` utility funtion to check the path of the data
    directory detected by GDAL (#160)

### Bug fixes

-   register GDAL drivers during initial import of pyogrio (#145)
-   support writing "not a time" (NaT) values in a datetime column (#146)
-   fixes an error when reading GPKG with bbox filter (#150)
-   properly raises error when invalid where clause is used on a GPKG (#150)
-   avoid duplicate count of available features (#151)

## 0.4.1 (2022-07-25)

### Bug fixes

-   use user-provided `encoding` when reading files instead of using default
    encoding of data source type (#139)
-   always convert curve or surface geometry types to linear geometry types,
    such as lines or polygons (#140)

## 0.4.0 (2022-06-20)

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
-   add `geometry_type` to `write_dataframe` to set geometry type for layer (#85)
-   Use certifi to set `GDAL_CURL_CA_BUNDLE` / `PROJ_CURL_CA_BUNDLE` defaults (#97)
-   automatically detect driver for `.geojson`, `.geojsonl` and `.geojsons` files (#101)
-   read DateTime fields with millisecond accuracy (#111)
-   support writing object columns with np.nan values (#118)
-   add support to write object columns that contain types different than string (#125)
-   support writing datetime columns (#120)
-   support for writing missing (null) geometries (#59)

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
    types if `geometry_type` is set to `"Unknown"`)
-   the geometry type of datasets with multiple geometry types will be set to
    `"Unknown"` unless overridden using `geometry_type`. Note:
    `"Unknown"` may be ignored by some drivers (e.g., shapefile)

### Bug fixes

-   use dtype `object` instead of `numpy.object` to eliminate deprecation warnings (#34)
-   raise error if layer cannot be opened (#35)
-   fix passing gdal creation parameters in `write_dataframe` (#62)
-   fix passing kwargs to GDAL in `write_dataframe` (#67)

### Changes from 0.4.0a1

-   `layer_geometry_type` introduced in 0.4.0a1 was renamed to `geometry_type` for consistency

### Contributors

People with a “+” by their names contributed a patch for the first time.

-   Brendan Ward
-   Joris Van den Bossche
-   Martin Fleischmann
-   Pieter Roggemans +
-   Wei Ji Leong +

## 0.3.0 (2021-12-22)

### Major enhancements

-   Auto-discovery of `GDAL_VERSION` on Windows, if `gdalinfo.exe` is discoverable
    on the `PATH`.
-   Addition of `read_bounds` function to read the bounds of each feature.
-   Addition of a `fids` keyword to `read` and `read_dataframe` to selectively
    read features based on a list of the FIDs.

## 0.2.0 (2021-04-02)

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

## 0.1.0 (2020-08-28)

### Major enhancements

-   Addition of `list_layers` to list layers in a data source.
-   Addition of `read_info` to read basic information for a layer.
-   Addition of `read_dataframe` to read from supported file formats (Shapefile, GeoPackage, GeoJSON) into GeoDataFrames.
-   Addition of `write_dataframe` to write GeoDataFrames into supported file formats.
