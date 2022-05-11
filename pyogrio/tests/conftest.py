from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pytest

from pyogrio import __gdal_version_string__, __version__, list_drivers
import pyogrio


_data_dir = Path(__file__).parent.resolve() / "fixtures"
ALL_EXTS = [".shp", ".gpkg", ".geojson", ".geojsons"]


def pytest_report_header(config):
    drivers = ", ".join(sorted(list(list_drivers(read=True).keys())))
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
    gdf = pyogrio.read_dataframe(testfile_path)
    pyogrio.write_dataframe(gdf, dst_path)
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
