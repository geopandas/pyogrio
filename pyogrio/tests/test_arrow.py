import contextlib
import json
import math
import os
import sys

import pytest
import numpy as np

import pyogrio
from pyogrio import __gdal_version__, read_dataframe, read_info, list_layers
from pyogrio.raw import open_arrow, read_arrow, write, write_arrow
from pyogrio.errors import DataSourceError, FieldError, DataLayerError
from pyogrio.tests.conftest import (
    ALL_EXTS,
    DRIVERS,
    DRIVER_EXT,
    requires_arrow_write_api,
    requires_pyarrow_api,
)

try:
    import pandas as pd
    from pandas.testing import assert_frame_equal, assert_index_equal
    from geopandas.testing import assert_geodataframe_equal

    import pyarrow
except ImportError:
    pass

# skip all tests in this file if Arrow API or GeoPandas are unavailable
pytestmark = requires_pyarrow_api
pytest.importorskip("geopandas")
pa = pytest.importorskip("pyarrow")


def test_read_arrow(naturalearth_lowres_all_ext):
    result = read_dataframe(naturalearth_lowres_all_ext, use_arrow=True)
    expected = read_dataframe(naturalearth_lowres_all_ext, use_arrow=False)

    if naturalearth_lowres_all_ext.suffix.startswith(".geojson"):
        check_less_precise = True
    else:
        check_less_precise = False
    assert_geodataframe_equal(result, expected, check_less_precise=check_less_precise)


def test_read_arrow_unspecified_layer_warning(data_dir):
    """Reading a multi-layer file without specifying a layer gives a warning."""
    with pytest.warns(UserWarning, match="More than one layer found "):
        read_arrow(data_dir / "sample.osm.pbf")


@pytest.mark.parametrize("skip_features, expected", [(10, 167), (200, 0)])
def test_read_arrow_skip_features(naturalearth_lowres, skip_features, expected):
    table = read_arrow(naturalearth_lowres, skip_features=skip_features)[1]
    assert len(table) == expected


def test_read_arrow_negative_skip_features(naturalearth_lowres):
    with pytest.raises(ValueError, match="'skip_features' must be >= 0"):
        read_arrow(naturalearth_lowres, skip_features=-1)


@pytest.mark.parametrize(
    "max_features, expected", [(0, 0), (10, 10), (200, 177), (100000, 177)]
)
def test_read_arrow_max_features(naturalearth_lowres, max_features, expected):
    table = read_arrow(naturalearth_lowres, max_features=max_features)[1]
    assert len(table) == expected


def test_read_arrow_negative_max_features(naturalearth_lowres):
    with pytest.raises(ValueError, match="'max_features' must be >= 0"):
        read_arrow(naturalearth_lowres, max_features=-1)


@pytest.mark.parametrize(
    "skip_features, max_features, expected",
    [
        (0, 0, 0),
        (10, 0, 0),
        (200, 0, 0),
        (1, 200, 176),
        (176, 10, 1),
        (100, 100, 77),
        (100, 100000, 77),
    ],
)
def test_read_arrow_skip_features_max_features(
    naturalearth_lowres, skip_features, max_features, expected
):
    table = read_arrow(
        naturalearth_lowres, skip_features=skip_features, max_features=max_features
    )[1]
    assert len(table) == expected


def test_read_arrow_fid(naturalearth_lowres_all_ext):
    kwargs = {"use_arrow": True, "where": "fid >= 2 AND fid <= 3"}

    df = read_dataframe(naturalearth_lowres_all_ext, fid_as_index=False, **kwargs)
    assert_index_equal(df.index, pd.RangeIndex(0, 2))

    df = read_dataframe(naturalearth_lowres_all_ext, fid_as_index=True, **kwargs)
    assert_index_equal(df.index, pd.Index([2, 3], name="fid"))


def test_read_arrow_columns(naturalearth_lowres):
    result = read_dataframe(naturalearth_lowres, use_arrow=True, columns=["continent"])
    assert result.columns.tolist() == ["continent", "geometry"]


def test_read_arrow_ignore_geometry(naturalearth_lowres):
    result = read_dataframe(naturalearth_lowres, use_arrow=True, read_geometry=False)
    assert type(result) is pd.DataFrame

    expected = read_dataframe(naturalearth_lowres, use_arrow=True).drop(
        columns=["geometry"]
    )
    assert_frame_equal(result, expected)


def test_read_arrow_nested_types(test_ogr_types_list):
    # with arrow, list types are supported
    result = read_dataframe(test_ogr_types_list, use_arrow=True)
    assert "list_int64" in result.columns
    assert result["list_int64"][0].tolist() == [0, 1]


def test_read_arrow_to_pandas_kwargs(test_fgdb_vsi):
    # with arrow, list types are supported
    arrow_to_pandas_kwargs = {"strings_to_categorical": True}
    result = read_dataframe(
        test_fgdb_vsi,
        layer="basetable_2",
        use_arrow=True,
        arrow_to_pandas_kwargs=arrow_to_pandas_kwargs,
    )
    assert "SEGMENT_NAME" in result.columns
    assert result["SEGMENT_NAME"].dtype.name == "category"


def test_read_arrow_raw(naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)
    assert isinstance(meta, dict)
    assert isinstance(table, pyarrow.Table)


def test_open_arrow_pyarrow(naturalearth_lowres):
    with open_arrow(naturalearth_lowres, use_pyarrow=True) as (meta, reader):
        assert isinstance(meta, dict)
        assert isinstance(reader, pyarrow.RecordBatchReader)
        assert isinstance(reader.read_all(), pyarrow.Table)


def test_open_arrow_batch_size(naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)
    batch_size = math.ceil(len(table) / 2)

    with open_arrow(naturalearth_lowres, batch_size=batch_size, use_pyarrow=True) as (
        meta,
        reader,
    ):
        assert isinstance(meta, dict)
        assert isinstance(reader, pyarrow.RecordBatchReader)
        count = 0
        tables = []
        for table in reader:
            tables.append(table)
            count += 1

        assert count == 2, "Should be two batches given the batch_size parameter"
        assert len(tables[0]) == batch_size, "First table should match the batch size"


@pytest.mark.skipif(
    __gdal_version__ >= (3, 8, 0),
    reason="skip_features supported by Arrow stream API for GDAL>=3.8.0",
)
@pytest.mark.parametrize("skip_features", [10, 200])
def test_open_arrow_skip_features_unsupported(naturalearth_lowres, skip_features):
    """skip_features are not supported for the Arrow stream interface for
    GDAL < 3.8.0"""
    with pytest.raises(
        ValueError,
        match="specifying 'skip_features' is not supported for Arrow for GDAL<3.8.0",
    ):
        with open_arrow(naturalearth_lowres, skip_features=skip_features) as (
            meta,
            reader,
        ):
            pass


@pytest.mark.parametrize("max_features", [10, 200])
def test_open_arrow_max_features_unsupported(naturalearth_lowres, max_features):
    """max_features are not supported for the Arrow stream interface"""
    with pytest.raises(
        ValueError,
        match="specifying 'max_features' is not supported for Arrow",
    ):
        with open_arrow(naturalearth_lowres, max_features=max_features) as (
            meta,
            reader,
        ):
            pass


@pytest.mark.skipif(
    __gdal_version__ < (3, 8, 0),
    reason="returns geoarrow metadata only for GDAL>=3.8.0",
)
def test_read_arrow_geoarrow_metadata(naturalearth_lowres):
    _meta, table = read_arrow(naturalearth_lowres)
    field = table.schema.field("wkb_geometry")
    assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"
    parsed_meta = json.loads(field.metadata[b"ARROW:extension:metadata"])
    assert parsed_meta["crs"]["id"]["authority"] == "EPSG"
    assert parsed_meta["crs"]["id"]["code"] == 4326


def test_open_arrow_capsule_protocol(naturalearth_lowres):
    pytest.importorskip("pyarrow", minversion="14")

    with open_arrow(naturalearth_lowres) as (meta, reader):
        assert isinstance(meta, dict)
        assert isinstance(reader, pyogrio._io._ArrowStream)

        result = pyarrow.table(reader)

    _, expected = read_arrow(naturalearth_lowres)
    assert result.equals(expected)


def test_open_arrow_capsule_protocol_without_pyarrow(naturalearth_lowres):
    pyarrow = pytest.importorskip("pyarrow", minversion="14")

    # Make PyArrow temporarily unavailable (importing will fail)
    sys.modules["pyarrow"] = None
    try:
        with open_arrow(naturalearth_lowres) as (meta, reader):
            assert isinstance(meta, dict)
            assert isinstance(reader, pyogrio._io._ArrowStream)
            result = pyarrow.table(reader)
    finally:
        sys.modules["pyarrow"] = pyarrow

    _, expected = read_arrow(naturalearth_lowres)
    assert result.equals(expected)


@contextlib.contextmanager
def use_arrow_context():
    original = os.environ.get("PYOGRIO_USE_ARROW", None)
    os.environ["PYOGRIO_USE_ARROW"] = "1"
    yield
    if original:
        os.environ["PYOGRIO_USE_ARROW"] = original
    else:
        del os.environ["PYOGRIO_USE_ARROW"]


def test_enable_with_environment_variable(test_ogr_types_list):
    # list types are only supported with arrow, so don't work by default and work
    # when arrow is enabled through env variable
    result = read_dataframe(test_ogr_types_list)
    assert "list_int64" not in result.columns

    with use_arrow_context():
        result = read_dataframe(test_ogr_types_list)
    assert "list_int64" in result.columns


@pytest.mark.skipif(
    __gdal_version__ < (3, 8, 3), reason="Arrow bool value bug fixed in GDAL >= 3.8.3"
)
@pytest.mark.parametrize("ext", ALL_EXTS)
def test_arrow_bool_roundtrip(tmpdir, ext):
    filename = os.path.join(str(tmpdir), f"test{ext}")

    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")] * 5, dtype=object
    )
    bool_col = np.array([True, False, True, False, True])
    field_data = [bool_col]
    fields = ["bool_col"]

    kwargs = {}

    if ext == ".fgb":
        # For .fgb, spatial_index=False to avoid the rows being reordered
        kwargs["spatial_index"] = False

    write(
        filename,
        geometry,
        field_data,
        fields,
        geometry_type="Point",
        crs="EPSG:4326",
        **kwargs,
    )

    write(
        filename, geometry, field_data, fields, geometry_type="Point", crs="EPSG:4326"
    )
    table = read_arrow(filename)[1]

    assert np.array_equal(table["bool_col"].to_numpy(), bool_col)


@pytest.mark.skipif(
    __gdal_version__ >= (3, 8, 3), reason="Arrow bool value bug fixed in GDAL >= 3.8.3"
)
@pytest.mark.parametrize("ext", ALL_EXTS)
def test_arrow_bool_exception(tmpdir, ext):
    filename = os.path.join(str(tmpdir), f"test{ext}")

    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")] * 5, dtype=object
    )
    bool_col = np.array([True, False, True, False, True])
    field_data = [bool_col]
    fields = ["bool_col"]

    write(
        filename, geometry, field_data, fields, geometry_type="Point", crs="EPSG:4326"
    )

    if ext in {".fgb", ".gpkg"}:
        # only raise exception for GPKG / FGB
        with pytest.raises(
            RuntimeError,
            match="GDAL < 3.8.3 does not correctly read boolean data values using "
            "the Arrow API",
        ):
            with open_arrow(filename):
                pass

        # do not raise exception if no bool columns are read
        with open_arrow(filename, columns=[]):
            pass

    else:
        with open_arrow(filename):
            pass


# Point(0, 0)
points = np.array(
    [bytes.fromhex("010100000000000000000000000000000000000000")] * 3,
    dtype=object,
)


@requires_arrow_write_api
def test_write_shp(tmpdir, naturalearth_lowres):
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
@requires_arrow_write_api
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
@requires_arrow_write_api
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


@requires_arrow_write_api
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


@pytest.mark.parametrize(
    "driver",
    {
        driver
        for driver in DRIVERS.values()
        if driver not in ("ESRI Shapefile", "GPKG", "GeoJSON")
    },
)
@requires_arrow_write_api
def test_write_supported(tmpdir, naturalearth_lowres, driver):
    """Test drivers known to work that are not specifically tested above"""
    meta, table = read_arrow(naturalearth_lowres, columns=["iso_a3"])

    # note: naturalearth_lowres contains mixed polygons / multipolygons, which
    # are not supported in mixed form for all drivers.  To get around this here
    # we take the first record only.
    meta["geometry_type"] = "MultiPolygon"

    filename = tmpdir / f"test{DRIVER_EXT[driver]}"
    write_arrow(
        table.slice(0, 1),
        filename,
        driver=driver,
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )
    assert filename.exists()


@requires_arrow_write_api
def test_write_unsupported(tmpdir, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.gdb")

    with pytest.raises(DataSourceError, match="does not support write functionality"):
        write_arrow(
            table,
            filename,
            driver="ESRIJSON",
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
        )


@pytest.mark.parametrize("ext", DRIVERS)
@requires_arrow_write_api
def test_write_append(request, tmpdir, naturalearth_lowres, ext):
    if ext.startswith(".geojson"):
        request.node.add_marker(
            pytest.mark.xfail(reason="Bugs with append when writing Arrow to GeoJSON")
        )

    meta, table = read_arrow(naturalearth_lowres)

    # coerce output layer to generic Geometry to avoid mixed type errors
    meta["geometry_type"] = "Unknown"

    filename = tmpdir / f"test{ext}"
    write_arrow(
        table,
        filename,
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )
    assert filename.exists()
    assert read_info(str(filename))["features"] == 177

    # write the same records again
    write_arrow(
        table,
        filename,
        append=True,
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )
    assert read_info(str(filename))["features"] == 354


@pytest.mark.parametrize("driver,ext", [("GML", ".gml"), ("GeoJSONSeq", ".geojsons")])
def test_write_append_unsupported(tmpdir, naturalearth_lowres, driver, ext):
    meta, table = read_arrow(naturalearth_lowres)

    # GML does not support append functionality
    filename = tmpdir / "test.gml"
    write_arrow(
        table,
        filename,
        driver="GML",
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )
    assert filename.exists()
    assert read_info(str(filename), force_feature_count=True)["features"] == 177

    with pytest.raises(DataSourceError):
        write_arrow(
            table,
            filename,
            driver="GML",
            append=True,
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
        )


@requires_arrow_write_api
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


@requires_arrow_write_api
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


@requires_arrow_write_api
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


@requires_arrow_write_api
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


@requires_arrow_write_api
def test_write_non_arrow_data(tmpdir):
    data = np.array([1, 2, 3])
    with pytest.raises(
        ValueError, match="The provided data is not recognized as Arrow data"
    ):
        write_arrow(
            data,
            tmpdir / "test_no_arrow_data.shp",
            crs="EPSG:4326",
            geometry_type="Point",
            geometry_name="geometry",
        )


@pytest.mark.filterwarnings("ignore:.*not handled natively:RuntimeWarning")
@requires_arrow_write_api
def test_write_batch_error_message(tmpdir):
    # raise the correct error and message from GDAL when an error happens
    # while writing

    # invalid dictionary array that will only error while writing (schema
    # itself is OK)
    arr = pa.DictionaryArray.from_buffers(
        pa.dictionary(pa.int64(), pa.string()),
        length=3,
        buffers=pa.array([0, 1, 2]).buffers(),
        dictionary=pa.array(["a", "b"]),
    )
    table = pa.table({"geometry": points, "col": arr})

    with pytest.raises(DataLayerError, match=".*invalid dictionary index"):
        write_arrow(
            table,
            tmpdir / "test_unsupported_list_type.fgb",
            crs="EPSG:4326",
            geometry_type="Point",
            geometry_name="geometry",
        )


@requires_arrow_write_api
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
