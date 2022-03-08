# Error handling

Pyogrio tries to capture and wrap errors from GDAL/OGR where possible, but defers
to error messages emitted by GDAL where available. The error types below are
intended to help assist in determining the source of the error in case the
error message is a bit cryptic.

Some of the errors that may be emitted by pyogrio include:

-   `ValueError` / `TypeError`: indicates that a user-provided is invalid for a particular
    operation
-   `DataSourceError`: indicates an error opening or using a transaction against a data source
-   `DataLayerError`: indicates an error obtaining a data layer or its properties (subclassed by all of following)
-   `CRSError`: indicates an error reading or writing CRS information
-   `FeatureError`: indicates an error reading or writing a specific feature
-   `GeometryError`: indicates an error reading or writing a geometry field of a single feature
-   `FieldError`: indicates an error reading or writing a non-geometry field of a single feature

All the pyogrio specific errors are subclasses of `RuntimeError`.
