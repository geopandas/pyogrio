
def get_gdal_version():
    """Convert GDAL version number into tuple of (major, minor, revision)"""
    version = int(GDALVersionInfo("VERSION_NUM"))
    major = version // 1000000
    minor = (version - (major * 1000000)) // 10000
    revision = (version - (major * 1000000) - (minor * 10000)) // 100
    return (major, minor, revision)