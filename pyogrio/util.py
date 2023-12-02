import re
import sys
from urllib.parse import urlparse

from packaging.version import Version

from pyogrio._env import GDALEnv

with GDALEnv():
    from pyogrio._ogr import buffer_to_virtual_file


def get_vsi_path(path_or_buffer):

    if hasattr(path_or_buffer, "read"):
        path_or_buffer = path_or_buffer.read()

    buffer = None
    if isinstance(path_or_buffer, bytes):
        buffer = path_or_buffer
        ext = ""
        is_zipped = path_or_buffer[:4].startswith(b"PK\x03\x04")
        if is_zipped:
            ext = ".zip"
        path = buffer_to_virtual_file(path_or_buffer, ext=ext)
        if is_zipped:
            path = "/vsizip/" + path
    else:
        path = vsi_path(str(path_or_buffer))

    return path, buffer


def vsi_path(path: str) -> str:
    """
    Ensure path is a local path or a GDAL-compatible vsi path.

    """

    # path is already in GDAL format
    if path.startswith("/vsi"):
        return path

    # Windows drive letters (e.g. "C:\") confuse `urlparse` as they look like
    # URL schemes
    if sys.platform == "win32" and re.match("^[a-zA-Z]\\:", path):
        if not path.split("!")[0].endswith(".zip"):
            return path

        # prefix then allow to proceed with remaining parsing
        path = f"zip://{path}"

    path, archive, scheme = _parse_uri(path)

    if scheme or archive or path.endswith(".zip"):
        return _construct_vsi_path(path, archive, scheme)

    return path


# Supported URI schemes and their mapping to GDAL's VSI suffix.
SCHEMES = {
    "file": "file",
    "zip": "zip",
    "tar": "tar",
    "gzip": "gzip",
    "http": "curl",
    "https": "curl",
    "ftp": "curl",
    "s3": "s3",
    "gs": "gs",
    "az": "az",
    "adls": "adls",
    "adl": "adls",  # fsspec uses this
    "hdfs": "hdfs",
    "webhdfs": "webhdfs",
    # GDAL additionally supports oss and swift for remote filesystems, but
    # those are for now not added as supported URI
}

CURLSCHEMES = set([k for k, v in SCHEMES.items() if v == "curl"])


def _parse_uri(path: str):
    """
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

    # if the scheme is not one of GDAL's supported schemes, return raw path
    if parts.scheme and not all(p in SCHEMES for p in parts.scheme.split("+")):
        return path, "", ""

    # we have a URI
    path = parts.path
    scheme = parts.scheme or ""

    if parts.query:
        path += "?" + parts.query

    if parts.scheme and parts.netloc:
        path = parts.netloc + path

    parts = path.split("!")
    path = parts.pop() if parts else ""
    archive = parts.pop() if parts else ""
    return (path, archive, scheme)


def _construct_vsi_path(path, archive, scheme) -> str:
    """Convert a parsed path to a GDAL VSI path"""

    prefix = ""
    suffix = ""
    schemes = scheme.split("+")

    if "zip" not in schemes and (archive.endswith(".zip") or path.endswith(".zip")):
        schemes.insert(0, "zip")

    if schemes:
        prefix = "/".join(
            "vsi{0}".format(SCHEMES[p]) for p in schemes if p and p != "file"
        )

        if schemes[-1] in CURLSCHEMES:
            suffix = f"{schemes[-1]}://"

    if prefix:
        if archive:
            return "/{}/{}{}/{}".format(prefix, suffix, archive, path.lstrip("/"))
        else:
            return "/{}/{}{}".format(prefix, suffix, path)

    return path


def _preprocess_options_key_value(options):
    """
    Preprocess options, eg `spatial_index=True` gets converted
    to `SPATIAL_INDEX="YES"`.
    """
    if not isinstance(options, dict):
        raise TypeError(f"Expected options to be a dict, got {type(options)}")

    result = {}
    for k, v in options.items():
        if v is None:
            continue
        k = k.upper()
        if isinstance(v, bool):
            v = "ON" if v else "OFF"
        else:
            v = str(v)
        result[k] = v
    return result


def _mask_to_wkb(mask):
    """Convert a Shapely mask geometry to WKB.

    Parameters
    ----------
    mask : Shapely geometry

    Returns
    -------
    WKB bytes or None

    Raises
    ------
    ValueError
        raised if Shapely >= 2.0 is not available or mask is not a Shapely
        Geometry object
    """

    if mask is None:
        return mask

    try:
        import shapely

        if Version(shapely.__version__) < Version("2.0.0"):
            shapely = None
    except ImportError:
        shapely = None

    if not shapely:
        raise ValueError("'mask' parameter requires Shapely >= 2.0")

    if not isinstance(mask, shapely.Geometry):
        raise ValueError("'mask' parameter must be a Shapely geometry")

    return shapely.to_wkb(mask)
