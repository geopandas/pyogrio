class DataSourceError(RuntimeError):
    """Errors relating to opening or closing an OGRDataSource (with >= 1 layers)"""

    pass


class DataLayerError(RuntimeError):
    """Errors relating to working with a single OGRLayer"""

    pass


class CRSError(DataLayerError):
    """Errors relating to getting or setting CRS values"""

    pass


class FeatureError(DataLayerError):
    """Errors related to reading or writing a feature"""

    pass


class GeometryError(DataLayerError):
    """Errors relating to getting or setting a geometry field"""

    pass


class FieldError(DataLayerError):
    """Errors relating to getting or setting a non-geometry field"""
