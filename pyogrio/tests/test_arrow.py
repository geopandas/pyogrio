import contextlib
import json
import math
import os
import sys
from io import BytesIO
from packaging.version import Version
from zipfile import ZipFile

import numpy as np

import pyogrio
from pyogrio import (
    __gdal_version__,
    get_gdal_config_option,
    list_layers,
    read_dataframe,
    read_info,
    set_gdal_config_options,
    vsi_listtree,
)
from pyogrio.errors import DataLayerError, DataSourceError, FieldError
from pyogrio.raw import open_arrow, read_arrow, write, write_arrow
from pyogrio.tests.conftest import (
    ALL_EXTS,
    DRIVER_EXT,
    DRIVERS,
    requires_arrow_write_api,
    requires_pyarrow_api,
)

import pytest

try:
    import pandas as pd
    import pyarrow

    from geopandas.testing import assert_geodataframe_equal
    from pandas.testing import assert_frame_equal, assert_index_equal
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


def test_read_arrow_nested_types(list_field_values_file):
    # with arrow, list types are supported
    result = read_dataframe(list_field_values_file, use_arrow=True)
    assert "list_int64" in result.columns
    assert result["list_int64"][0].tolist() == [0, 1]


def test_read_arrow_to_pandas_kwargs(no_geometry_file):
    # with arrow, list types are supported
    arrow_to_pandas_kwargs = {"strings_to_categorical": True}
    df = read_dataframe(
        no_geometry_file,
        read_geometry=False,
        use_arrow=True,
        arrow_to_pandas_kwargs=arrow_to_pandas_kwargs,
    )
    assert df.col.dtype.name == "category"
    assert np.array_equal(df.col.values.categories, ["a", "b", "c"])


def test_read_arrow_raw(naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)
    assert isinstance(meta, dict)
    assert isinstance(table, pyarrow.Table)


def test_read_arrow_vsi(naturalearth_lowres_vsi):
    table = read_arrow(naturalearth_lowres_vsi[1])[1]
    assert len(table) == 177

    # Check temp file was cleaned up. Filter to files created by pyogrio, as GDAL keeps
    # cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


def test_read_arrow_bytes(geojson_bytes):
    meta, table = read_arrow(geojson_bytes)

    assert meta["fields"].shape == (5,)
    assert len(table) == 3

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


def test_read_arrow_nonseekable_bytes(nonseekable_bytes):
    meta, table = read_arrow(nonseekable_bytes)
    assert meta["fields"].shape == (0,)
    assert len(table) == 1

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


def test_read_arrow_filelike(geojson_filelike):
    meta, table = read_arrow(geojson_filelike)

    assert meta["fields"].shape == (5,)
    assert len(table) == 3

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


def test_open_arrow_pyarrow(naturalearth_lowres):
    with open_arrow(naturalearth_lowres, use_pyarrow=True) as (meta, reader):
        assert isinstance(meta, dict)
        assert isinstance(reader, pyarrow.RecordBatchReader)
        assert isinstance(reader.read_all(), pyarrow.Table)


def test_open_arrow_batch_size(naturalearth_lowres):
    _, table = read_arrow(naturalearth_lowres)
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


def test_enable_with_environment_variable(list_field_values_file):
    # list types are only supported with arrow, so don't work by default and work
    # when arrow is enabled through env variable
    result = read_dataframe(list_field_values_file)
    assert "list_int64" not in result.columns

    with use_arrow_context():
        result = read_dataframe(list_field_values_file)

    assert "list_int64" in result.columns


@pytest.mark.skipif(
    __gdal_version__ < (3, 8, 3), reason="Arrow bool value bug fixed in GDAL >= 3.8.3"
)
@pytest.mark.parametrize("ext", ALL_EXTS)
def test_arrow_bool_roundtrip(tmp_path, ext):
    filename = tmp_path / f"test{ext}"

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
def test_arrow_bool_exception(tmp_path, ext):
    filename = tmp_path / f"test{ext}"

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
def test_write_shp(tmp_path, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    filename = tmp_path / "test.shp"
    write_arrow(
        table,
        filename,
        crs=meta["crs"],
        encoding=meta["encoding"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert filename.exists()
    for ext in (".dbf", ".prj"):
        assert filename.with_suffix(ext).exists()


@pytest.mark.filterwarnings("ignore:A geometry of type POLYGON is inserted")
@requires_arrow_write_api
def test_write_gpkg(tmp_path, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    filename = tmp_path / "test.gpkg"
    write_arrow(
        table,
        filename,
        driver="GPKG",
        crs=meta["crs"],
        geometry_type="MultiPolygon",
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert filename.exists()


@pytest.mark.filterwarnings("ignore:A geometry of type POLYGON is inserted")
@requires_arrow_write_api
def test_write_gpkg_multiple_layers(tmp_path, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)
    meta["geometry_type"] = "MultiPolygon"

    filename = tmp_path / "test.gpkg"
    write_arrow(
        table,
        filename,
        driver="GPKG",
        layer="first",
        crs=meta["crs"],
        geometry_type="MultiPolygon",
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert filename.exists()

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
def test_write_geojson(tmp_path, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)
    filename = tmp_path / "test.json"
    write_arrow(
        table,
        filename,
        driver="GeoJSON",
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert filename.exists()

    data = json.loads(open(filename).read())

    assert data["type"] == "FeatureCollection"
    assert data["name"] == "test"
    assert "crs" in data
    assert len(data["features"]) == len(table)
    assert not len(
        set(meta["fields"]).difference(data["features"][0]["properties"].keys())
    )


@requires_arrow_write_api
@pytest.mark.skipif(
    __gdal_version__ < (3, 6, 0),
    reason="OpenFileGDB write support only available for GDAL >= 3.6.0",
)
@pytest.mark.parametrize(
    "write_int64",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                __gdal_version__ < (3, 9, 0),
                reason="OpenFileGDB write support for int64 values for GDAL >= 3.9.0",
            ),
        ),
    ],
)
def test_write_openfilegdb(tmp_path, write_int64):
    expected_field_data = [
        np.array([True, False, True], dtype="bool"),
        np.array([1, 2, 3], dtype="int16"),
        np.array([1, 2, 3], dtype="int32"),
        np.array([1, 2, 3], dtype="int64"),
        np.array([1, 2, 3], dtype="float32"),
        np.array([1, 2, 3], dtype="float64"),
    ]

    table = pa.table(
        {
            "geometry": points,
            **{field.dtype.name: field for field in expected_field_data},
        }
    )

    filename = tmp_path / "test.gdb"

    expected_meta = {"crs": "EPSG:4326"}

    # int64 is not supported without additional config: https://gdal.org/en/latest/drivers/vector/openfilegdb.html#bit-integer-field-support
    # it is converted to float64 by default and raises a warning
    # (for GDAL >= 3.9.0 only)
    write_params = (
        {"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"} if write_int64 else {}
    )

    if write_int64 or __gdal_version__ < (3, 9, 0):
        ctx = contextlib.nullcontext()
    else:
        ctx = pytest.warns(
            RuntimeWarning, match="Integer64 will be written as a Float64"
        )

    with ctx:
        write_arrow(
            table,
            filename,
            driver="OpenFileGDB",
            geometry_type="Point",
            geometry_name="geometry",
            **expected_meta,
            **write_params,
        )

    meta, table = read_arrow(filename)

    if not write_int64:
        expected_field_data[3] = expected_field_data[3].astype("float64")

    # bool types are converted to int32
    expected_field_data[0] = expected_field_data[0].astype("int32")

    assert meta["crs"] == expected_meta["crs"]

    # NOTE: geometry name is set to "SHAPE" by GDAL
    assert np.array_equal(table[meta["geometry_name"]], points)
    for i in range(len(expected_field_data)):
        values = table[table.schema.names[i]].to_numpy()
        assert values.dtype == expected_field_data[i].dtype
        assert np.array_equal(values, expected_field_data[i])


@pytest.mark.parametrize(
    "driver",
    {
        driver
        for driver in DRIVERS.values()
        if driver not in ("ESRI Shapefile", "GPKG", "GeoJSON")
    },
)
@requires_arrow_write_api
def test_write_supported(tmp_path, naturalearth_lowres, driver):
    """Test drivers known to work that are not specifically tested above"""
    meta, table = read_arrow(naturalearth_lowres, columns=["iso_a3"], max_features=1)

    # note: naturalearth_lowres contains mixed polygons / multipolygons, which
    # are not supported in mixed form for all drivers.  To get around this here
    # we take the first record only.
    meta["geometry_type"] = "MultiPolygon"

    filename = tmp_path / f"test{DRIVER_EXT[driver]}"
    write_arrow(
        table,
        filename,
        driver=driver,
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )
    assert filename.exists()


@requires_arrow_write_api
def test_write_unsupported(tmp_path, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    with pytest.raises(DataSourceError, match="does not support write functionality"):
        write_arrow(
            table,
            tmp_path / "test.json",
            driver="ESRIJSON",
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
        )


@pytest.mark.parametrize("ext", DRIVERS)
@requires_arrow_write_api
def test_write_append(request, tmp_path, naturalearth_lowres, ext):
    if ext.startswith(".geojson"):
        # Bug in GDAL when appending int64 to GeoJSON
        # (https://github.com/OSGeo/gdal/issues/9792)
        request.node.add_marker(
            pytest.mark.xfail(reason="Bugs with append when writing Arrow to GeoJSON")
        )

    meta, table = read_arrow(naturalearth_lowres)

    # coerce output layer to generic Geometry to avoid mixed type errors
    meta["geometry_type"] = "Unknown"

    filename = tmp_path / f"test{ext}"
    write_arrow(
        table,
        filename,
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )
    assert filename.exists()
    assert read_info(filename)["features"] == 177

    # write the same records again
    write_arrow(
        table,
        filename,
        append=True,
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )
    assert read_info(filename)["features"] == 354


@pytest.mark.parametrize("driver,ext", [("GML", ".gml"), ("GeoJSONSeq", ".geojsons")])
@requires_arrow_write_api
def test_write_append_unsupported(tmp_path, naturalearth_lowres, driver, ext):
    meta, table = read_arrow(naturalearth_lowres)

    # GML does not support append functionality
    filename = tmp_path / "test.gml"
    write_arrow(
        table,
        filename,
        driver="GML",
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )
    assert filename.exists()
    assert read_info(filename, force_feature_count=True)["features"] == 177

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
def test_write_gdalclose_error(naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    filename = "s3://non-existing-bucket/test.geojson"

    # set config options to avoid errors on open due to GDAL S3 configuration
    set_gdal_config_options(
        {
            "AWS_ACCESS_KEY_ID": "invalid",
            "AWS_SECRET_ACCESS_KEY": "invalid",
            "AWS_NO_SIGN_REQUEST": True,
        }
    )

    with pytest.raises(DataSourceError, match="Failed to write features to dataset"):
        write_arrow(
            table,
            filename,
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
        )


@requires_arrow_write_api
@pytest.mark.parametrize("name", ["geoarrow.wkb", "ogc.wkb"])
def test_write_geometry_extension_type(tmp_path, naturalearth_lowres, name):
    # Infer geometry column based on extension name
    # instead of passing `geometry_name` explicitly
    meta, table = read_arrow(naturalearth_lowres)

    # change extension type name
    idx = table.schema.get_field_index("wkb_geometry")
    new_field = table.schema.field(idx).with_metadata({"ARROW:extension:name": name})
    new_table = table.cast(table.schema.set(idx, new_field))

    filename = tmp_path / "test_geoarrow.shp"
    write_arrow(
        new_table,
        filename,
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
    )
    _, table_roundtripped = read_arrow(filename)
    assert table_roundtripped.equals(table)


@requires_arrow_write_api
def test_write_unsupported_geoarrow(tmp_path, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    # change extension type name (the name doesn't match with the column type
    # for correct geoarrow data, but our writing code checks it based on the name)
    idx = table.schema.get_field_index("wkb_geometry")
    new_field = table.schema.field(idx).with_metadata(
        {"ARROW:extension:name": "geoarrow.point"}
    )
    new_table = table.cast(table.schema.set(idx, new_field))

    with pytest.raises(
        NotImplementedError,
        match="Writing a geometry column of type geoarrow.point is not yet supported",
    ):
        write_arrow(
            new_table,
            tmp_path / "test_geoarrow.shp",
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
        )


@requires_arrow_write_api
def test_write_no_geom(tmp_path, naturalearth_lowres):
    _, table = read_arrow(naturalearth_lowres)
    table = table.drop_columns("wkb_geometry")

    # Test
    filename = tmp_path / "test.gpkg"
    write_arrow(table, filename)
    # Check result
    assert filename.exists()
    meta, result = read_arrow(filename)
    assert meta["crs"] is None
    assert meta["geometry_type"] is None
    assert table.equals(result)


@requires_arrow_write_api
def test_write_geometry_type(tmp_path, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    # Not specifying the geometry currently raises an error
    with pytest.raises(ValueError, match="'geometry_type' keyword is required"):
        write_arrow(
            table,
            tmp_path / "test.shp",
            crs=meta["crs"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
        )

    # Specifying "Unknown" works and will create generic layer
    filename = tmp_path / "test.gpkg"
    write_arrow(
        table,
        filename,
        crs=meta["crs"],
        geometry_type="Unknown",
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )
    assert filename.exists()
    meta_written, _ = read_arrow(filename)
    assert meta_written["geometry_type"] == "Unknown"


@requires_arrow_write_api
def test_write_raise_promote_to_multi(tmp_path, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    with pytest.raises(
        ValueError, match="The 'promote_to_multi' option is not supported"
    ):
        write_arrow(
            table,
            tmp_path / "test.shp",
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
            promote_to_multi=True,
        )


@requires_arrow_write_api
def test_write_no_crs(tmp_path, naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)

    filename = tmp_path / "test.shp"
    with pytest.warns(UserWarning, match="'crs' was not provided"):
        write_arrow(
            table,
            filename,
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
        )
    # apart from CRS warning, it did write correctly
    meta_result, result = read_arrow(filename)
    assert table.equals(result)
    assert meta_result["crs"] is None


@requires_arrow_write_api
def test_write_non_arrow_data(tmp_path):
    data = np.array([1, 2, 3])
    with pytest.raises(
        ValueError, match="The provided data is not recognized as Arrow data"
    ):
        write_arrow(
            data,
            tmp_path / "test_no_arrow_data.shp",
            crs="EPSG:4326",
            geometry_type="Point",
            geometry_name="geometry",
        )


@pytest.mark.skipif(
    Version(pa.__version__) < Version("16.0.0.dev0"),
    reason="PyCapsule protocol only added to pyarrow.ChunkedArray in pyarrow 16",
)
@requires_arrow_write_api
def test_write_non_arrow_tabular_data(tmp_path):
    data = pa.chunked_array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(
        DataLayerError,
        match=".*should be called on a schema that is a struct of fields",
    ):
        write_arrow(
            data,
            tmp_path / "test_no_arrow_tabular_data.shp",
            crs="EPSG:4326",
            geometry_type="Point",
            geometry_name="geometry",
        )


@pytest.mark.filterwarnings("ignore:.*not handled natively:RuntimeWarning")
@requires_arrow_write_api
def test_write_batch_error_message(tmp_path):
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
            tmp_path / "test_unsupported_list_type.fgb",
            crs="EPSG:4326",
            geometry_type="Point",
            geometry_name="geometry",
        )


@requires_arrow_write_api
def test_write_schema_error_message(tmp_path):
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
            tmp_path / "test_unsupported_map_type.shp",
            crs="EPSG:4326",
            geometry_type="Point",
            geometry_name="geometry",
        )


@requires_arrow_write_api
@pytest.mark.filterwarnings("ignore:File /vsimem:RuntimeWarning")
@pytest.mark.parametrize("driver", ["GeoJSON", "GPKG"])
def test_write_memory(naturalearth_lowres, driver):
    meta, table = read_arrow(naturalearth_lowres, max_features=1)
    meta["geometry_type"] = "MultiPolygon"

    buffer = BytesIO()
    write_arrow(
        table,
        buffer,
        driver=driver,
        layer="test",
        crs=meta["crs"],
        geometry_type=meta["geometry_type"],
        geometry_name=meta["geometry_name"] or "wkb_geometry",
    )

    assert len(buffer.getbuffer()) > 0
    assert list_layers(buffer)[0][0] == "test"

    actual_meta, actual_table = read_arrow(buffer)
    assert len(actual_table) == len(table)
    assert np.array_equal(actual_meta["fields"], meta["fields"])


@requires_arrow_write_api
def test_write_memory_driver_required(naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres, max_features=1)

    buffer = BytesIO()
    with pytest.raises(
        ValueError,
        match="driver must be provided to write to in-memory file",
    ):
        write_arrow(
            table,
            buffer,
            driver=None,
            layer="test",
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
        )

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


@requires_arrow_write_api
@pytest.mark.parametrize("driver", ["ESRI Shapefile", "OpenFileGDB"])
def test_write_memory_unsupported_driver(naturalearth_lowres, driver):
    if driver == "OpenFileGDB" and __gdal_version__ < (3, 6, 0):
        pytest.skip("OpenFileGDB write support only available for GDAL >= 3.6.0")

    meta, table = read_arrow(naturalearth_lowres, max_features=1)

    buffer = BytesIO()

    with pytest.raises(
        ValueError, match=f"writing to in-memory file is not supported for {driver}"
    ):
        write_arrow(
            table,
            buffer,
            driver=driver,
            layer="test",
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
        )


@requires_arrow_write_api
@pytest.mark.parametrize("driver", ["GeoJSON", "GPKG"])
def test_write_memory_append_unsupported(naturalearth_lowres, driver):
    meta, table = read_arrow(naturalearth_lowres, max_features=1)
    meta["geometry_type"] = "MultiPolygon"

    buffer = BytesIO()
    with pytest.raises(
        NotImplementedError, match="append is not supported for in-memory files"
    ):
        write_arrow(
            table,
            buffer,
            driver=driver,
            layer="test",
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
            append=True,
        )


@requires_arrow_write_api
def test_write_memory_existing_unsupported(naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres, max_features=1)
    meta["geometry_type"] = "MultiPolygon"

    buffer = BytesIO(b"0000")
    with pytest.raises(
        NotImplementedError,
        match="writing to existing in-memory object is not supported",
    ):
        write_arrow(
            table,
            buffer,
            driver="GeoJSON",
            layer="test",
            crs=meta["crs"],
            geometry_type=meta["geometry_type"],
            geometry_name=meta["geometry_name"] or "wkb_geometry",
        )


@requires_arrow_write_api
def test_write_open_file_handle(tmp_path, naturalearth_lowres):
    """Verify that writing to an open file handle is not currently supported"""

    meta, table = read_arrow(naturalearth_lowres, max_features=1)
    meta["geometry_type"] = "MultiPolygon"

    # verify it fails for regular file handle
    with pytest.raises(
        NotImplementedError, match="writing to an open file handle is not yet supported"
    ):
        with open(tmp_path / "test.geojson", "wb") as f:
            write_arrow(
                table,
                f,
                driver="GeoJSON",
                layer="test",
                crs=meta["crs"],
                geometry_type=meta["geometry_type"],
                geometry_name=meta["geometry_name"] or "wkb_geometry",
            )

    # verify it fails for ZipFile
    with pytest.raises(
        NotImplementedError, match="writing to an open file handle is not yet supported"
    ):
        with ZipFile(tmp_path / "test.geojson.zip", "w") as z:
            with z.open("test.geojson", "w") as f:
                write_arrow(
                    table,
                    f,
                    driver="GeoJSON",
                    layer="test",
                    crs=meta["crs"],
                    geometry_type=meta["geometry_type"],
                    geometry_name=meta["geometry_name"] or "wkb_geometry",
                )

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


@requires_arrow_write_api
def test_non_utf8_encoding_io_shapefile(tmp_path, encoded_text):
    encoding, text = encoded_text

    table = pa.table(
        {
            # Point(0, 0)
            "geometry": pa.array(
                [bytes.fromhex("010100000000000000000000000000000000000000")]
            ),
            text: pa.array([text]),
        }
    )

    filename = tmp_path / "test.shp"
    write_arrow(
        table,
        filename,
        geometry_type="Point",
        geometry_name="geometry",
        crs="EPSG:4326",
        encoding=encoding,
    )

    # NOTE: GDAL automatically creates a cpg file with the encoding name, which
    # means that if we read this without specifying the encoding it uses the
    # correct one
    schema, table = read_arrow(filename)
    assert schema["fields"][0] == text
    assert table[text][0].as_py() == text

    # verify that if cpg file is not present, that user-provided encoding must be used
    filename.with_suffix(".cpg").unlink()

    # We will assume ISO-8859-1, which is wrong
    miscoded = text.encode(encoding).decode("ISO-8859-1")
    bad_schema = read_arrow(filename)[0]
    assert bad_schema["fields"][0] == miscoded
    # table cannot be decoded to UTF-8 without UnicodeDecodeErrors

    # If encoding is provided, that should yield correct text
    schema, table = read_arrow(filename, encoding=encoding)
    assert schema["fields"][0] == text
    assert table[text][0].as_py() == text

    # verify that setting encoding does not corrupt SHAPE_ENCODING option if set
    # globally (it is ignored during read when encoding is specified by user)
    try:
        set_gdal_config_options({"SHAPE_ENCODING": "CP1254"})
        _ = read_arrow(filename, encoding=encoding)
        assert get_gdal_config_option("SHAPE_ENCODING") == "CP1254"

    finally:
        # reset to clear between tests
        set_gdal_config_options({"SHAPE_ENCODING": None})


@requires_arrow_write_api
def test_encoding_write_layer_option_collision_shapefile(tmp_path, naturalearth_lowres):
    """Providing both encoding parameter and ENCODING layer creation option
    (even if blank) is not allowed."""

    meta, table = read_arrow(naturalearth_lowres)

    with pytest.raises(
        ValueError,
        match=(
            'cannot provide both encoding parameter and "ENCODING" layer creation '
            "option"
        ),
    ):
        write_arrow(
            table,
            tmp_path / "test.shp",
            crs=meta["crs"],
            geometry_type="MultiPolygon",
            geometry_name=meta["geometry_name"] or "wkb_geometry",
            encoding="CP936",
            layer_options={"ENCODING": ""},
        )


@requires_arrow_write_api
@pytest.mark.parametrize("ext", ["gpkg", "geojson"])
def test_non_utf8_encoding_io_arrow_exception(tmp_path, naturalearth_lowres, ext):
    meta, table = read_arrow(naturalearth_lowres)

    with pytest.raises(
        ValueError, match="non-UTF-8 encoding is not supported for Arrow"
    ):
        write_arrow(
            table,
            tmp_path / f"test.{ext}",
            crs=meta["crs"],
            geometry_type="MultiPolygon",
            geometry_name=meta["geometry_name"] or "wkb_geometry",
            encoding="CP936",
        )
