from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pytest

from pyogrio import __gdal_version_string__, __version__, list_drivers
from pyogrio.raw import read, write


_data_dir = Path(__file__).parent.resolve() / "fixtures"
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
        for ext in ["dbf", "prj", "shp", "shx"]:
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
