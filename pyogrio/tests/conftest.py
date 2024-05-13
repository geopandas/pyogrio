from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from pyogrio import (
    __gdal_version_string__,
    __version__,
    list_drivers,
)
from pyogrio._compat import (
    HAS_ARROW_API,
    HAS_ARROW_WRITE_API,
    HAS_GDAL_GEOS,
    HAS_PYARROW,
    HAS_SHAPELY,
)
from pyogrio.raw import read, write

_data_dir = Path(__file__).parent.resolve() / "fixtures"

# mapping of driver extension to driver name for well-supported drivers
DRIVERS = {
    ".fgb": "FlatGeobuf",
    ".geojson": "GeoJSON",
    ".geojsonl": "GeoJSONSeq",
    ".geojsons": "GeoJSONSeq",
    ".gpkg": "GPKG",
    ".shp": "ESRI Shapefile",
}

# mapping of driver name to extension
DRIVER_EXT = {driver: ext for ext, driver in DRIVERS.items()}

ALL_EXTS = [".fgb", ".geojson", ".geojsonl", ".gpkg", ".shp"]


def pytest_report_header(config):
    drivers = ", ".join(
        f"{driver}({capability})"
        for driver, capability in sorted(list_drivers().items())
    )
    return (
        f"pyogrio {__version__}\n"
        f"GDAL {__gdal_version_string__}\n"
        f"Supported drivers: {drivers}"
    )


# marks to skip tests if optional dependecies are not present
requires_arrow_api = pytest.mark.skipif(not HAS_ARROW_API, reason="GDAL>=3.6 required")
requires_pyarrow_api = pytest.mark.skipif(
    not HAS_ARROW_API or not HAS_PYARROW, reason="GDAL>=3.6 and pyarrow required"
)

requires_arrow_write_api = pytest.mark.skipif(
    not HAS_ARROW_WRITE_API or not HAS_PYARROW,
    reason="GDAL>=3.8 required for Arrow write API",
)

requires_gdal_geos = pytest.mark.skipif(
    not HAS_GDAL_GEOS, reason="GDAL compiled with GEOS required"
)

requires_shapely = pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely >= 2.0 required")


def prepare_testfile(testfile_path, dst_dir, ext):
    if ext == testfile_path.suffix:
        return testfile_path

    dst_path = dst_dir / f"{testfile_path.stem}{ext}"
    if dst_path.exists():
        return dst_path

    meta, _, geometry, field_data = read(testfile_path)

    if ext == ".fgb":
        # For .fgb, spatial_index=False to avoid the rows being reordered
        meta["spatial_index"] = False
        # allow mixed Polygons/MultiPolygons type
        meta["geometry_type"] = "Unknown"

    elif ext == ".gpkg":
        # For .gpkg, spatial_index=False to avoid the rows being reordered
        meta["spatial_index"] = False
        meta["geometry_type"] = "MultiPolygon"

    write(dst_path, geometry, field_data, **meta)
    return dst_path


@pytest.fixture(scope="session")
def data_dir():
    return _data_dir


@pytest.fixture(scope="function")
def naturalearth_lowres(tmp_path, request):
    ext = getattr(request, "param", ".shp")
    testfile_path = _data_dir / Path("naturalearth_lowres/naturalearth_lowres.shp")

    return prepare_testfile(testfile_path, tmp_path, ext)


@pytest.fixture(scope="function", params=ALL_EXTS)
def naturalearth_lowres_all_ext(tmp_path, naturalearth_lowres, request):
    return prepare_testfile(naturalearth_lowres, tmp_path, request.param)


@pytest.fixture(scope="function")
def naturalearth_lowres_vsi(tmp_path, naturalearth_lowres):
    """Wrap naturalearth_lowres as a zip file for vsi tests"""

    path = tmp_path / f"{naturalearth_lowres.name}.zip"
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED, compresslevel=5) as out:
        for ext in ["dbf", "prj", "shp", "shx", "cpg"]:
            filename = f"{naturalearth_lowres.stem}.{ext}"
            out.write(naturalearth_lowres.parent / filename, filename)

    return path, f"/vsizip/{path}/{naturalearth_lowres.name}"


@pytest.fixture(scope="session")
def test_fgdb_vsi():
    return f"/vsizip/{_data_dir}/test_fgdb.gdb.zip"


@pytest.fixture(scope="session")
def test_gpkg_nulls():
    return _data_dir / "test_gpkg_nulls.gpkg"


@pytest.fixture(scope="session")
def test_ogr_types_list():
    return _data_dir / "test_ogr_types_list.geojson"


@pytest.fixture(scope="session")
def test_datetime():
    return _data_dir / "test_datetime.geojson"


@pytest.fixture(scope="session")
def test_datetime_tz():
    return _data_dir / "test_datetime_tz.geojson"


@pytest.fixture(scope="function")
def geojson_bytes(tmp_path):
    """Extracts first 3 records from naturalearth_lowres and writes to GeoJSON,
    returning bytes"""
    meta, _, geometry, field_data = read(
        _data_dir / Path("naturalearth_lowres/naturalearth_lowres.shp"), max_features=3
    )

    filename = tmp_path / "test.geojson"
    write(filename, geometry, field_data, **meta)

    with open(filename, "rb") as f:
        bytes_buffer = f.read()

    return bytes_buffer


@pytest.fixture(scope="function")
def geojson_filelike(tmp_path):
    """Extracts first 3 records from naturalearth_lowres and writes to GeoJSON,
    returning open file handle"""
    meta, _, geometry, field_data = read(
        _data_dir / Path("naturalearth_lowres/naturalearth_lowres.shp"), max_features=3
    )

    filename = tmp_path / "test.geojson"
    write(filename, geometry, field_data, layer="test", **meta)

    with open(filename, "rb") as f:
        yield f


@pytest.fixture(
    scope="session",
    params=[
        # Japanese
        ("CP932", "ﾎ"),
        # Chinese
        ("CP936", "中文"),
        # Central European
        ("CP1250", "Đ"),
        # Latin 1 / Western European
        ("CP1252", "ÿ"),
        # Greek
        ("CP1253", "Φ"),
        # Arabic
        ("CP1256", "ش"),
    ],
)
def encoded_text(request):
    """Return tuple with encoding name and very short sample text in that encoding
    NOTE: it was determined through testing that code pages for MS-DOS do not
    consistently work across all Python installations (in particular, fail with conda),
    but ANSI code pages appear to work properly.
    """
    return request.param
