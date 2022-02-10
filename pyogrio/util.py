"""Dataset paths, identifiers, and filenames"""

import re
import sys
from urllib.parse import urlparse


def vsi_path(path: str) -> str:
    """
    Ensure path is a local path or a GDAL-compatible vsi path.

    """
    # Windows drive letters (e.g. "C:\") confuse `urlparse` as they look like
    # URL schemes
    if sys.platform == "win32" and re.match("^[a-zA-Z]\\:", path):
        return path

    elif path.startswith("/vsi"):
        return path

    elif re.match("^[a-z0-9\\+]*://", path):

        path, archive, scheme = _parse_uri(path)
        if not scheme:
            return path

        return _construct_vsi_path(path, archive, scheme)

    else:
        return path


# Supported URI schemes and their mapping to GDAL's VSI suffix.
SCHEMES = {
    "ftp": "curl",
    "gzip": "gzip",
    "http": "curl",
    "https": "curl",
    "s3": "s3",
    "tar": "tar",
    "zip": "zip",
    "file": "file",
    "gs": "gs",
    "az": "az",
    "adls": "adls",
    "adl": "adls",  # fsspec uses this
    # 'oss': 'oss',
    # 'swift': 'swift',
    "hdfs": "hdfs",
    "webhdfs": "webhdfs",
}

CURLSCHEMES = set([k for k, v in SCHEMES.items() if v == "curl"])


def _parse_uri(path: str):
    """"
    Parse a URI

    Returns a tuples of (path, archive, scheme)

    path : str
        Parsed path. Includes the hostname and query string in the case
        of a URI.
    archive : str
        Parsed archive path.
    scheme : str
        URI scheme such as "https" or "zip+s3".
    """
    parts = urlparse(path)

    # if the scheme is not one of Rasterio's supported schemes, we
    # return an UnparsedPath.
    if parts.scheme and not all(p in SCHEMES for p in parts.scheme.split("+")):
        return path

    # we have a URI
    path = parts.path
    scheme = parts.scheme or None

    if parts.query:
        path += "?" + parts.query

    if parts.scheme and parts.netloc:
        path = parts.netloc + path

    parts = path.split("!")
    path = parts.pop() if parts else None
    archive = parts.pop() if parts else None
    return (path, archive, scheme)


def _construct_vsi_path(path, archive, scheme) -> str:
    """Convert a parsed path to a GDAL VSI path"""

    if scheme.split("+")[-1] in CURLSCHEMES:
        suffix = "{}://".format(scheme.split("+")[-1])
    else:
        suffix = ""

    prefix = "/".join(
        "vsi{0}".format(SCHEMES[p]) for p in scheme.split("+") if p != "file"
    )

    if prefix:
        if archive:
            result = "/{}/{}{}/{}".format(prefix, suffix, archive, path.lstrip("/"))
        else:
            result = "/{}/{}{}".format(prefix, suffix, path)
    else:
        result = path

    return result
