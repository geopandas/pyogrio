import json
import os

import numpy as np
import pytest

from pyogrio.core import list_layers
from pyogrio.raw import read_arrow, write_arrow
from pyogrio.tests.conftest import requires_arrow_write_api

try:
    import pandas as pd
    import pyarrow
    from geopandas.testing import assert_geodataframe_equal
    from pandas.testing import assert_frame_equal, assert_index_equal
except ImportError:
    pass

# skip all tests in this file if Arrow Write API or GeoPandas are unavailable
pytestmark = requires_arrow_write_api
pytest.importorskip("geopandas")


def test_write(tmpdir, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.shp")
    write_arrow(
        filename,
        table,
        crs=meta["crs"],
        encoding=meta["encoding"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert os.path.exists(filename)
    for ext in (".dbf", ".prj"):
        assert os.path.exists(filename.replace(".shp", ext))


def test_write_gpkg(tmpdir, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.gpkg")
    write_arrow(
        filename,
        table,
        driver="GPKG",
        crs=meta["crs"],
        encoding=meta["encoding"],
        geometry_type="MultiPolygon",
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert os.path.exists(filename)


def test_write_gpkg_multiple_layers(tmpdir, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)
    meta["geometry_type"] = "MultiPolygon"

    filename = os.path.join(str(tmpdir), "test.gpkg")
    write_arrow(
        filename,
        table,
        driver="GPKG",
        layer="first",
        crs=meta["crs"],
        encoding=meta["encoding"],
        geometry_type="MultiPolygon",
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert os.path.exists(filename)

    assert np.array_equal(list_layers(filename), [["first", "MultiPolygon"]])

    write_arrow(
        filename,
        table,
        driver="GPKG",
        layer="second",
        crs=meta["crs"],
        encoding=meta["encoding"],
        geometry_type="MultiPolygon",
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert np.array_equal(
        list_layers(filename), [["first", "MultiPolygon"], ["second", "MultiPolygon"]]
    )


def test_write_geojson(tmpdir, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.json")
    write_arrow(
        filename,
        table,
        driver="GeoJSON",
        crs=meta["crs"],
        encoding=meta["encoding"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert os.path.exists(filename)

    data = json.loads(open(filename).read())

    assert data["type"] == "FeatureCollection"
    assert data["name"] == "test"
    assert "crs" in data
    assert len(data["features"]) == len(table)
    assert not len(
        set(meta["fields"]).difference(data["features"][0]["properties"].keys())
    )
