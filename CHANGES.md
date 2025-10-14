# CHANGELOG

## 0.12.0 (xxxx-xx-xx)

### Potentially breaking changes

-   Return JSON fields (as identified by GDAL) as dicts/lists in `read_dataframe`;
    these were previously returned as strings (#556).
-   Drop support for GDAL 3.4 and 3.5 (#584).

### Improvements

-   Add listing of GDAL data types and subtypes to `read_info` (#556).
-   Add support to read list fields without arrow (#558).

### Bug fixes

-   Fix decode error reading an sqlite file on windows (#568).
-   Fix wrong layername when creating .gpkg.zip file (#570).
-   Fix segfault on providing an invalid value for `layer` in `read_info` (#564).

### Packaging

-   The GDAL library included in the wheels is upgraded from 3.10.3 to 3.11.4 (#578).
-   Add libkml driver to the wheels for more recent Linux platforms supported
    by manylinux_2_28, MacOS, and Windows (#561).
-   Minimum required Python version is now 3.10 (#557).
-   Wheels are now available for Python 3.14 (#579).

## 0.11.1 (2025-08-02)

### Bug fixes

-   Compatibility with Shapely >= 2.1 to avoid triggering a deprecation warning at
    import (#542).
-   Fix reading with a `skip_features` larger than the available number of
    features to ensure this consistently returns an empty result for all file
    formats (#550).

## 0.11.0 (2025-05-08)

### Improvements

-   Capture all errors logged by gdal when opening a file fails (#495).
-   Add support to read and write ".gpkg.zip" (GDAL >= 3.7), ".shp.zip", and ".shz"
    files (#527).
-   Compatibility with the string dtype in the upcoming pandas 3.0 release (#493).

### Bug fixes

-   Fix WKB writing on big-endian systems (#497).
-   Fix writing fids to e.g. GPKG file with `use_arrow` (#511).
-   Fix error in `write_dataframe` when writing an empty or all-None object
    column with `use_arrow` (#512).

### Packaging

-   The GDAL library included in the wheels is upgraded from 3.9.2 to 3.10.3 (#499).

## 0.10.0 (2024-09-28)

### Improvements

-   Add support to read, write, list, and remove `/vsimem/` files (#457).
-   Raise specific error when trying to read non-UTF-8 file with
    `use_arrow=True` (#490).

### Bug fixes

-   Silence warning from `write_dataframe` with `GeoSeries.notna()` (#435).
-   Enable mask & bbox filter when geometry column not read (#431).
-   Raise `NotImplementedError` when user attempts to write to an open file handle (#442).
-   Prevent seek on read from compressed inputs (#443).

### Packaging

-   For the conda-forge package, change the dependency from `libgdal` to
    `libgdal-core`. This package is significantly smaller as it doesn't contain
    some large GDAL plugins. Extra plugins can be installed as seperate conda
    packages if needed: more info [here](https://gdal.org/download.html#conda).
    This also leads to `pyproj` becoming an optional dependency; you will need
    to install `pyproj` in order to support spatial reference systems (#452).
-   The GDAL library included in the wheels is updated from 3.8.5 to GDAL 3.9.2 (#466).
-   pyogrio now requires a minimum version of Python >= 3.9 (#473).
-   Wheels are now available for Python 3.13.

## 0.9.0 (2024-06-17)

### Improvements

-   Add `on_invalid` parameter to `read_dataframe` (#422).

### Bug fixes

-   Fixed bug transposing longitude and latitude when writing files with
    coordinate transformation from EPSG:4326 (#421).
-   Fix bug preventing reading from file paths containing hashes in `read_dataframe` (#412).

### Packaging

-   MacOS wheels are now only available for macOS 12+. For older unsupported macOS
    versions, pyogrio can still be built from source (requires GDAL to be installed) (#417).
-   Remove usage of deprecated `distutils` in `setup.py` (#416).

## 0.8.0 (2024-05-06)

### Improvements

-   Support for writing based on Arrow as the transfer mechanism of the data
    from Python to GDAL (requires GDAL >= 3.8). This is provided through the
    new `pyogrio.raw.write_arrow` function, or by using the `use_arrow=True`
    option in `pyogrio.write_dataframe` (#314, #346).
-   Add support for `fids` filter to `read_arrow` and `open_arrow`, and to
    `read_dataframe` with `use_arrow=True` (#304).
-   Add some missing properties to `read_info`, including layer name, geometry name
    and FID column name (#365).
-   `read_arrow` and `open_arrow` now provide
    [GeoArrow-compliant extension metadata](https://geoarrow.org/extension-types.html),
    including the CRS, when using GDAL 3.8 or higher (#366).
-   The `open_arrow` function can now be used without a `pyarrow` dependency. By
    default, it will now return a stream object implementing the
    [Arrow PyCapsule Protocol](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
    (i.e. having an `__arrow_c_stream__`method). This object can then be consumed
    by your Arrow implementation of choice that supports this protocol. To keep
    the previous behaviour of returning a `pyarrow.RecordBatchReader`, specify
    `use_pyarrow=True` (#349).
-   Warn when reading from a multilayer file without specifying a layer (#362).
-   Allow writing to a new in-memory datasource using io.BytesIO object (#397).

### Bug fixes

-   Fix error in `write_dataframe` if input has a date column and
    non-consecutive index values (#325).
-   Fix encoding issues on windows for some formats (e.g. ".csv") and always write ESRI
    Shapefiles using UTF-8 by default on all platforms (#361).
-   Raise exception in `read_arrow` or `read_dataframe(..., use_arrow=True)` if
    a boolean column is detected due to error in GDAL reading boolean values for
    FlatGeobuf / GPKG drivers (#335, #387); this has been fixed in GDAL >= 3.8.3.
-   Properly ignore fields not listed in `columns` parameter when reading from
    the data source not using the Arrow API (#391).
-   Properly handle decoding of ESRI Shapefiles with user-provided `encoding`
    option for `read`, `read_dataframe`, and `open_arrow`, and correctly encode
    Shapefile field names and text values to the user-provided `encoding` for
    `write` and `write_dataframe` (#384).
-   Fixed bug preventing reading from bytes or file-like in `read_arrow` /
    `open_arrow` (#407).

### Packaging

-   The GDAL library included in the wheels is updated from 3.7.2 to GDAL 3.8.5.

### Potentially breaking changes

-   Using a `where` expression combined with a list of `columns` that does not include
    the column referenced in the expression is not recommended and will now
    return results based on driver-dependent behavior, which may include either
    returning empty results (even if non-empty results are expected from `where` parameter)
    or raise an exception (#391). Previous versions of pyogrio incorrectly
    set ignored fields against the data source, allowing it to return non-empty
    results in these cases.

## 0.7.2 (2023-10-30)

### Bug fixes

-   Add `packaging` as a dependency (#320).
-   Fix conversion of WKB to geometries with missing values when using
    `pandas.ArrowDtype` (#321).

## 0.7.1 (2023-10-26)

### Bug fixes

-   Fix unspecified dependency on `packaging` (#318).

## 0.7.0 (2023-10-25)

### Improvements

-   Support reading and writing datetimes with timezones (#253).
-   Support writing dataframes without geometry column (#267).
-   Calculate feature count by iterating over features if GDAL returns an
    unknown count for a data layer (e.g., OSM driver); this may have signficant
    performance impacts for some data sources that would otherwise return an
    unknown count (count is used in `read_info`, `read`, `read_dataframe`) (#271).
-   Add `arrow_to_pandas_kwargs` parameter to `read_dataframe` + reduce memory usage
    with `use_arrow=True` (#273)
-   In `read_info`, the result now also contains the `total_bounds` of the layer as well
    as some extra `capabilities` of the data source driver (#281).
-   Raise error if `read` or `read_dataframe` is called with parameters to read no
    columns, geometry, or fids (#280).
-   Automatically detect supported driver by extension for all available
    write drivers and addition of `detect_write_driver` (#270).
-   Addition of `mask` parameter to `open_arrow`, `read`, `read_dataframe`,
    and `read_bounds` functions to select only the features in the dataset that
    intersect the mask geometry (#285). Note: GDAL < 3.8.0 returns features that
    intersect the bounding box of the mask when using the Arrow interface for
    some drivers; this has been fixed in GDAL 3.8.0.
-   Removed warning when no features are read from the data source (#299).
-   Add support for `force_2d=True` with `use_arrow=True` in `read_dataframe` (#300).

### Other changes

-   test suite requires Shapely >= 2.0

-   using `skip_features` greater than the number of features available in a data
    layer now returns empty arrays for `read` and an empty DataFrame for
    `read_dataframe` instead of raising a `ValueError` (#282).
-   enabled `skip_features` and `max_features` for `read_arrow` and
    `read_dataframe(path, use_arrow=True)`. Note that this incurs overhead
    because all features up to the next batch size above `max_features` (or size
    of data layer) will be read prior to slicing out the requested range of
    features (#282).
-   The `use_arrow=True` option can be enabled globally for testing using the
    `PYOGRIO_USE_ARROW=1` environment variable (#296).

### Bug fixes

-   Fix int32 overflow when reading int64 columns (#260)
-   Fix `fid_as_index=True` doesn't set fid as index using `read_dataframe` with
    `use_arrow=True` (#265)
-   Fix errors reading OSM data due to invalid feature count and incorrect
    reading of OSM layers beyond the first layer (#271)
-   Always raise an exception if there is an error when writing a data source
    (#284)

### Potentially breaking changes

-   In `read_info` (#281):
    -   the `features` property in the result will now be -1 if calculating the
        feature count is an expensive operation for this driver. You can force it to be
        calculated using the `force_feature_count` parameter.
    -   for boolean values in the `capabilities` property, the values will now be
        booleans instead of 1 or 0.

### Packaging

-   The GDAL library included in the wheels is updated from 3.6.4 to GDAL 3.7.2.

## 0.6.0 (2023-04-27)

### Improvements

-   Add automatic detection of 3D geometries in `write_dataframe` (#223, #229)
-   Add "driver" property to `read_info` result (#224)
-   Add support for dataset open options to `read`, `read_dataframe`, and
    `read_info` (#233)
-   Add support for pandas' nullable data types in `write_dataframe`, or
    specifying a mask manually for missing values in `write` (#219)
-   Standardized 3-dimensional geometry type labels from "2.5D <type>" to
    "<type> Z" for consistency with well-known text (WKT) formats (#234)
-   Failure error messages from GDAL are no longer printed to stderr (they were
    already translated into Python exceptions as well) (#236).
-   Failure and warning error messages from GDAL are no longer printed to
    stderr: failures were already translated into Python exceptions
    and warning messages are now translated into Python warnings (#236, #242).
-   Add access to low-level pyarrow `RecordBatchReader` via
    `pyogrio.raw.open_arrow`, which allows iterating over batches of Arrow
    tables (#205).
-   Add support for writing dataset and layer metadata (where supported by
    driver) to `write` and `write_dataframe`, and add support for reading
    dataset and layer metadata in `read_info` (#237).

### Packaging

-   The GDAL library included in the wheels is updated from 3.6.2 to GDAL 3.6.4.
-   Wheels are now available for Linux aarch64 / arm64.

## 0.5.1 (2023-01-26)

### Bug fixes

-   Fix memory leak in reading files (#207)
-   Fix to only use transactions for writing records when supported by the
    driver (#203)

## 0.5.0 (2023-01-16)

### Major enhancements

-   Support for reading based on Arrow as the transfer mechanism of the data
    from GDAL to Python (requires GDAL >= 3.6 and `pyarrow` to be installed).
    This can be enabled by passing `use_arrow=True` to `pyogrio.read_dataframe`
    (or by using `pyogrio.raw.read_arrow` directly), and provides a further
    speed-up (#155, #191).
-   Support for appending to an existing data source when supported by GDAL by
    passing `append=True` to `pyogrio.write_dataframe` (#197).

### Potentially breaking changes

-   In floating point columns, NaN values are now by default written as "null"
    instead of NaN, but with an option to control this (pass `nan_as_null=False`
    to keep the previous behaviour) (#190).

### Improvements

-   It is now possible to pass GDAL's dataset creation options in addition
    to layer creation options in `pyogrio.write_dataframe` (#189).
-   When specifying a subset of `columns` to read, unnecessary IO or parsing
    is now avoided (#195).

### Packaging

-   The GDAL library included in the wheels is updated from 3.4 to GDAL 3.6.2,
    and is now built with GEOS and sqlite with rtree support enabled
    (which allows writing a spatial index for GeoPackage).
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
