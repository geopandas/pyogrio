import os
import contextlib

import pytest

import pyogrio
import pyogrio.raw
from pyogrio.util import vsi_path


@contextlib.contextmanager
def change_cwd(path):
    curdir = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(curdir)


@pytest.mark.parametrize(
    "path, expected",
    [
        # local file paths that should be passed through as is
        ("data.gpkg", "data.gpkg"),
        ("/home/user/data.gpkg", "/home/user/data.gpkg"),
        (r"C:\User\Documents\data.gpkg", r"C:\User\Documents\data.gpkg"),
        ("file:///home/user/data.gpkg", "/home/user/data.gpkg"),
        # cloud URIs
        ("s3://testing/data.gpkg", "/vsis3/testing/data.gpkg"),
        ("gs://testing/data.gpkg", "/vsigs/testing/data.gpkg"),
        ("az://testing/data.gpkg", "/vsiaz/testing/data.gpkg"),
        ("adl://testing/data.gpkg", "/vsiadls/testing/data.gpkg"),
        ("adls://testing/data.gpkg", "/vsiadls/testing/data.gpkg"),
        # archives
        ("zip://data.zip", "/vsizip/data.zip"),
        ("tar://data.tar", "/vsitar/data.tar"),
        ("gzip://data.gz", "/vsigzip/data.gz"),
        ("tar://./my.tar!my.geojson", "/vsitar/./my.tar/my.geojson"),
        (
            "zip://home/data/shapefile.zip!layer.shp",
            "/vsizip/home/data/shapefile.zip/layer.shp",
        ),
        # combined schemes
        ("zip+s3://testing/shapefile.zip", "/vsizip/vsis3/testing/shapefile.zip"),
        (
            "zip+https://s3.amazonaws.com/testing/shapefile.zip",
            "/vsizip/vsicurl/https://s3.amazonaws.com/testing/shapefile.zip",
        ),
    ],
)
def test_vsi_path(path, expected):
    assert vsi_path(path) == expected


def test_vsi_handling_read_functions(naturalearth_lowres_vsi):
    # test that all different read entry points have the path handling
    # (a zip:// path would otherwise fail)
    path, _ = naturalearth_lowres_vsi
    path = "zip://" + str(path)

    result = pyogrio.raw.read(path)
    assert len(result[1]) == 177

    result = pyogrio.read_info(path)
    assert result["features"] == 177

    result = pyogrio.read_bounds(path)
    assert len(result[0]) == 177

    result = pyogrio.read_dataframe(path)
    assert len(result) == 177


def test_path_absolute(data_dir):
    # pathlib path
    path = data_dir / "naturalearth_lowres/naturalearth_lowres.shp"
    df = pyogrio.read_dataframe(path)
    len(df) == 177

    # str path
    df = pyogrio.read_dataframe(str(path))
    len(df) == 177


def test_path_relative(data_dir):
    with change_cwd(data_dir):
        df = pyogrio.read_dataframe("naturalearth_lowres/naturalearth_lowres.shp")
    len(df) == 177


def test_uri_local_file(data_dir):
    uri = "file://" + str(data_dir / "naturalearth_lowres/naturalearth_lowres.shp")
    df = pyogrio.read_dataframe(uri)
    len(df) == 177


def test_zip_path(naturalearth_lowres_vsi):
    path, path_vsi = naturalearth_lowres_vsi
    path_zip = "zip://" + str(path)

    # absolute zip path
    df = pyogrio.read_dataframe(path_zip)
    assert len(df) == 177

    # relative zip path
    with change_cwd(path.parent):
        df = pyogrio.read_dataframe("zip://" + path.name)
    assert len(df) == 177

    # absolute vsizip path
    df = pyogrio.read_dataframe(path_vsi)
    assert len(df) == 177


@pytest.mark.network
def test_url():
    df = pyogrio.read_dataframe(
        "https://raw.githubusercontent.com/geopandas/pyogrio/main/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"
    )
    assert len(df) == 177


@pytest.mark.network
def test_url_with_zip():
    df = pyogrio.read_dataframe(
        "zip+https://s3.amazonaws.com/fiona-testing/coutwildrnp.zip"
    )
    assert len(df) == 67


# @pytest.mark.network
# def test_uri_s3()
#     df = pyogrio.read_dataframe('zip+s3://fiona-testing/coutwildrnp.zip')
#     assert len(df) == 67
