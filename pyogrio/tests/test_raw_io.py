import json
import os

import numpy as np
from numpy import array_equal
import pytest

from pyogrio import list_layers
from pyogrio.raw import read, write


def test_read(naturalearth_lowres):
    meta, geometry, fields = read(naturalearth_lowres)

    assert meta["crs"] == "EPSG:4326"
    assert meta["geometry_type"] == "Polygon"
    assert meta["encoding"] == "UTF-8"
    assert meta["fields"].shape == (5,)

    assert meta["fields"].tolist() == [
        "pop_est",
        "continent",
        "name",
        "iso_a3",
        "gdp_md_est",
    ]

    assert len(fields) == 5
    assert len(geometry) == len(fields[0])

    # quick test that WKB is a Polygon type
    assert geometry[0][:6] == b"\x01\x06\x00\x00\x00\x03"


def test_vsi_read_layers(naturalearth_lowres_vsi):
    assert array_equal(
        list_layers(naturalearth_lowres_vsi), [["naturalearth_lowres", "Polygon"]]
    )

    meta, geometry, fields = read(naturalearth_lowres_vsi)
    assert geometry.shape == (177,)


def test_read_no_geometry(naturalearth_lowres):
    meta, geometry, fields = read(naturalearth_lowres, read_geometry=False)

    assert geometry is None


def test_read_columns(naturalearth_lowres):
    # read no columns or geometry
    meta, geometry, fields = read(naturalearth_lowres, columns=[], read_geometry=False)
    assert geometry is None
    assert len(fields) == 0
    array_equal(meta["fields"], np.empty(shape=(0, 4), dtype="object"))

    columns = ["NAME", "NAME_LONG"]
    meta, geometry, fields = read(
        naturalearth_lowres, columns=columns, read_geometry=False
    )
    array_equal(meta["fields"], columns)

    # Repeats should be dropped
    columns = ["NAME", "NAME_LONG", "NAME"]
    meta, geometry, fields = read(
        naturalearth_lowres, columns=columns, read_geometry=False
    )
    array_equal(meta["fields"], columns[:2])


def test_read_skip_features(naturalearth_lowres):
    expected_geometry, expected_fields = read(naturalearth_lowres)[1:]
    geometry, fields = read(naturalearth_lowres, skip_features=10)[1:]

    assert len(geometry) == len(expected_geometry) - 10
    assert len(fields[0]) == len(expected_fields[0]) - 10

    assert np.array_equal(geometry, expected_geometry[10:])
    # Last field has more variable data
    assert np.array_equal(fields[-1], expected_fields[-1][10:])


def test_read_max_features(naturalearth_lowres):
    expected_geometry, expected_fields = read(naturalearth_lowres)[1:]
    geometry, fields = read(naturalearth_lowres, max_features=2)[1:]

    assert len(geometry) == 2
    assert len(fields[0]) == 2

    assert np.array_equal(geometry, expected_geometry[:2])
    assert np.array_equal(fields[-1], expected_fields[-1][:2])


def test_read_where(naturalearth_lowres):
    # empty filter should return full set of records
    geometry, fields = read(naturalearth_lowres, where="")[1:]
    assert len(geometry) == 177
    assert len(fields) == 5
    assert len(fields[0]) == 177

    # should return singular item
    geometry, fields = read(naturalearth_lowres, where="iso_a3 = 'CAN'")[1:]
    assert len(geometry) == 1
    assert len(fields) == 5
    assert len(fields[0]) == 1
    assert fields[3] == "CAN"

    # should return items within range
    geometry, fields = read(
        naturalearth_lowres, where="POP_EST >= 10000000 AND POP_EST < 100000000"
    )[1:]
    assert len(geometry) == 75
    assert min(fields[0]) >= 10000000
    assert max(fields[0]) < 100000000

    # should match no items
    with pytest.warns(UserWarning, match="does not have any features to read"):
        geometry, fields = read(naturalearth_lowres, where="iso_a3 = 'INVALID'")[1:]
        assert len(geometry) == 0


def test_read_where_invalid(naturalearth_lowres):
    with pytest.raises(ValueError, match="Invalid SQL"):
        read(naturalearth_lowres, where="invalid")


@pytest.mark.parametrize("bbox", [(1,), (1, 2), (1, 2, 3)])
def test_read_bbox_invalid(naturalearth_lowres, bbox):
    with pytest.raises(ValueError, match="Invalid bbox"):
        read(naturalearth_lowres, bbox=bbox)


def test_read_bbox(naturalearth_lowres):
    # should return no features
    with pytest.warns(UserWarning, match="does not have any features to read"):
        geometry, fields = read(naturalearth_lowres, bbox=(0, 0, 0.00001, 0.00001))[1:]

    assert len(geometry) == 0

    geometry, fields = read(naturalearth_lowres, bbox=(-140, 20, -100, 40))[1:]

    assert len(geometry) == 2
    assert np.array_equal(fields[3], ["USA", "MEX"])


def test_write(tmpdir, naturalearth_lowres):
    meta, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.shp")
    write(filename, geometry, field_data, **meta)

    assert os.path.exists(filename)
    for ext in (".dbf", ".prj"):
        assert os.path.exists(filename.replace(".shp", ext))


def test_write_gpkg(tmpdir, naturalearth_lowres):
    meta, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.gpkg")
    write(filename, geometry, field_data, driver="GPKG", **meta)

    assert os.path.exists(filename)


def test_write_geojson(tmpdir, naturalearth_lowres):
    meta, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.json")
    write(filename, geometry, field_data, driver="GeoJSON", **meta)

    assert os.path.exists(filename)

    data = json.loads(open(filename).read())

    assert data["type"] == "FeatureCollection"
    assert data["name"] == "test"
    assert "crs" in data
    assert len(data["features"]) == len(geometry)
    assert not len(
        set(meta["fields"]).difference(data["features"][0]["properties"].keys())
    )


def test_write_geojsonseq(tmpdir, naturalearth_lowres):
    meta, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.json")
    write(filename, geometry, field_data, driver="GeoJSONSeq", **meta)

    assert os.path.exists(filename)

