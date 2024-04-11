import json
import os

import numpy as np
import pytest

from pyogrio.core import list_layers
from pyogrio.raw import read_arrow, write_arrow
from pyogrio.tests.conftest import requires_arrow_write_api
from pyogrio.errors import FieldError


# skip all tests in this file if Arrow Write API or GeoPandas are unavailable
pytestmark = requires_arrow_write_api
pytest.importorskip("geopandas")
pa = pytest.importorskip("pyarrow")


# Point(0, 0)
points = np.array(
    [bytes.fromhex("010100000000000000000000000000000000000000")] * 3,
    dtype=object,
)


def test_write(tmpdir, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.shp")
    write_arrow(
        table,
        filename,
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
        table,
        filename,
        driver="GPKG",
        crs=meta["crs"],
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
        table,
        filename,
        driver="GPKG",
        layer="first",
        crs=meta["crs"],
        geometry_type="MultiPolygon",
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert os.path.exists(filename)

    assert np.array_equal(list_layers(filename), [["first", "MultiPolygon"]])

    write_arrow(
        table,
        filename,
        driver="GPKG",
        layer="second",
        crs=meta["crs"],
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
        table,
        filename,
        driver="GeoJSON",
        crs=meta["crs"],
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
        new_table,
        filename,
        crs=meta["crs"],
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
            new_table,
            filename,
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
        )


def test_write_geometry_type(tmpdir, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    # Not specifying the geometry currently raises an error
    filename = os.path.join(str(tmpdir), "test.shp")
    with pytest.raises(ValueError, match="Need to specify 'geometry_type"):
        write_arrow(
            table,
            filename,
            crs=meta["crs"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
        )

    # Specifying "Unknown" works and will create generic layer
    filename = os.path.join(str(tmpdir), "test.gpkg")
    write_arrow(
        table,
        filename,
        crs=meta["crs"],
        geometry_type="Unknown",
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )
    assert os.path.exists(filename)
    meta_written, _ = read_arrow(filename)
    assert meta_written["geometry_type"] == "Unknown"


def test_write_raise_promote_to_multi(tmpdir, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.shp")

    with pytest.raises(
        ValueError, match="The 'promote_to_multi' option is not supported"
    ):
        write_arrow(
            table,
            filename,
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
            promote_to_multi=True,
        )


@pytest.mark.filterwarnings("ignore:.*not handled natively:RuntimeWarning")
def test_write_batch_error_message(tmpdir):
    # raise the correct error and message from GDAL when an error happens
    # while writing
    table = pa.table({"geometry": points, "col": [[0, 1], [2, 3, 4], [4]]})

    with pytest.raises(
        Exception, match="ICreateFeature: Missing implementation for OGRFieldType 13"
    ):
        write_arrow(
            table,
            tmpdir / "test_unsupported_list_type.fgb",
            crs="EPSG:4326",
            geometry_type="Point",
            geometry_name="geometry",
        )


def test_write_schema_error_message(tmpdir):
    # raise the correct error and message from GDAL when an error happens
    # creating the fields from the schema
    # (using complex list of map of integer->integer which is not supported by GDAL)
    table = pa.table(
        {
            "geometry": points,
            "col": pa.array(
                [[[(1, 2), (3, 4)], None, [(5, 6)]]] * 3,
                pa.list_(pa.map_(pa.int64(), pa.int64())),
            ),
        }
    )

    with pytest.raises(FieldError, match=".*not supported"):
        write_arrow(
            table,
            tmpdir / "test_unsupported_map_type.shp",
            crs="EPSG:4326",
            geometry_type="Point",
            geometry_name="geometry",
        )
