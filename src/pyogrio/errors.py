class CRSError(Exception):
    pass


class DriverError(Exception):
    pass


class TransactionError(RuntimeError):
    pass


class UnsupportedGeometryTypeError(Exception):
    pass


class DriverIOError(IOError):
    pass
