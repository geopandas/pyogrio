import json
import os

import numpy as np
import pytest

from pyogrio.core import list_layers
from pyogrio.raw import read_arrow, write_arrow
from pyogrio.tests.conftest import requires_arrow_write_api


# skip all tests in this file if Arrow Write API or GeoPandas are unavailable
pytestmark = requires_arrow_write_api
pytest.importorskip("geopandas")
pytest.importorskip("pyarrow")


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


@pytest.mark.filterwarnings("ignore:A geometry of type POLYGON is inserted")
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


@pytest.mark.filterwarnings("ignore:A geometry of type POLYGON is inserted")
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


@pytest.mark.parametrize("name", ["geoarrow.wkb", "ogc.wkb"])
def test_write_geometry_extension_type(tmpdir, naturalearth_lowres, name):
    # Infer geometry column based on extension name
    # instead of passing `geometry_name` explicitly
    meta, table = read_arrow(naturalearth_lowres)

    # change extension type name
    idx = table.schema.get_field_index("wkb_geometry")
    new_field = table.schema.field(idx).with_metadata({"ARROW:extension:name": name})
    new_table = table.cast(table.schema.set(idx, new_field))

    filename = os.path.join(str(tmpdir), "test_geoarrow.shp")
    write_arrow(
        filename,
        new_table,
        crs=meta["crs"],
        encoding=meta["encoding"],
        geometry_type=meta["geometry_type"],
    )
    _, table_roundtripped = read_arrow(filename)
    assert table_roundtripped.equals(table)


def test_write_unsupported_geoarrow(tmpdir, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    # change extension type name (the name doesn't match with the column type
    # for correct geoarrow data, but our writing code checks it based on the name)
    idx = table.schema.get_field_index("wkb_geometry")
    new_field = table.schema.field(idx).with_metadata(
        {"ARROW:extension:name": "geoarrow.point"}
    )
    new_table = table.cast(table.schema.set(idx, new_field))

    filename = os.path.join(str(tmpdir), "test_geoarrow.shp")
    with pytest.raises(
        NotImplementedError,
        match="Writing a geometry column of type geoarrow.point is not yet supported",
    ):
        write_arrow(
            filename,
            new_table,
            crs=meta["crs"],
            encoding=meta["encoding"],
            geometry_type=meta["geometry_type"],
        )
