class CRSError(Exception):
    pass


class DriverError(Exception):
    pass


class UnsupportedGeometryTypeError(Exception):
    pass


class DriverIOError(IOError):
    pass
