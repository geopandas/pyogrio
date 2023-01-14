import json
import os
import sys

import numpy as np
from numpy import array_equal
import pytest

from pyogrio import list_layers, list_drivers, read_info, __gdal_version__
from pyogrio.raw import DRIVERS, read, write
from pyogrio.errors import DataSourceError, DataLayerError, FeatureError
from pyogrio.tests.conftest import prepare_testfile

# mapping of driver name to extension
DRIVER_EXT = {driver: ext for ext, driver in DRIVERS.items()}


def test_read(naturalearth_lowres):
    meta, _, geometry, fields = read(naturalearth_lowres)

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


@pytest.mark.parametrize("ext", DRIVERS)
def test_read_autodetect_driver(tmp_path, naturalearth_lowres, ext):
    # Test all supported autodetect drivers
    testfile = prepare_testfile(naturalearth_lowres, dst_dir=tmp_path, ext=ext)

    assert testfile.suffix == ext
    assert testfile.exists()
    meta, _, geometry, fields = read(testfile)

    assert meta["crs"] == "EPSG:4326"
    assert meta["geometry_type"] in ("MultiPolygon", "Polygon", "Unknown")
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


def test_read_invalid_layer(naturalearth_lowres):
    with pytest.raises(DataLayerError, match="Layer 'invalid' could not be opened"):
        read(naturalearth_lowres, layer="invalid")

    with pytest.raises(DataLayerError, match="Layer '-1' could not be opened"):
        read(naturalearth_lowres, layer=-1)

    with pytest.raises(DataLayerError, match="Layer '2' could not be opened"):
        read(naturalearth_lowres, layer=2)


def test_vsi_read_layers(naturalearth_lowres_vsi):
    _, naturalearth_lowres_vsi = naturalearth_lowres_vsi
    assert array_equal(
        list_layers(naturalearth_lowres_vsi), [["naturalearth_lowres", "Polygon"]]
    )

    geometry = read(naturalearth_lowres_vsi)[2]
    assert geometry.shape == (177,)


def test_read_no_geometry(naturalearth_lowres):
    geometry = read(naturalearth_lowres, read_geometry=False)[2]

    assert geometry is None


def test_read_columns(naturalearth_lowres):
    # read no columns or geometry
    meta, _, geometry, fields = read(
        naturalearth_lowres, columns=[], read_geometry=False
    )
    assert geometry is None
    assert len(fields) == 0
    array_equal(meta["fields"], np.empty(shape=(0, 4), dtype="object"))

    columns = ["NAME", "NAME_LONG"]
    meta, _, geometry, fields = read(
        naturalearth_lowres, columns=columns, read_geometry=False
    )
    array_equal(meta["fields"], columns)

    # Repeats should be dropped
    columns = ["NAME", "NAME_LONG", "NAME"]
    meta, _, geometry, fields = read(
        naturalearth_lowres, columns=columns, read_geometry=False
    )
    array_equal(meta["fields"], columns[:2])


def test_read_skip_features(naturalearth_lowres):
    expected_geometry, expected_fields = read(naturalearth_lowres)[2:]
    geometry, fields = read(naturalearth_lowres, skip_features=10)[2:]

    assert len(geometry) == len(expected_geometry) - 10
    assert len(fields[0]) == len(expected_fields[0]) - 10

    assert np.array_equal(geometry, expected_geometry[10:])
    # Last field has more variable data
    assert np.array_equal(fields[-1], expected_fields[-1][10:])


def test_read_max_features(naturalearth_lowres):
    expected_geometry, expected_fields = read(naturalearth_lowres)[2:]
    geometry, fields = read(naturalearth_lowres, max_features=2)[2:]

    assert len(geometry) == 2
    assert len(fields[0]) == 2

    assert np.array_equal(geometry, expected_geometry[:2])
    assert np.array_equal(fields[-1], expected_fields[-1][:2])


def test_read_where(naturalearth_lowres):
    # empty filter should return full set of records
    geometry, fields = read(naturalearth_lowres, where="")[2:]
    assert len(geometry) == 177
    assert len(fields) == 5
    assert len(fields[0]) == 177

    # should return singular item
    geometry, fields = read(naturalearth_lowres, where="iso_a3 = 'CAN'")[2:]
    assert len(geometry) == 1
    assert len(fields) == 5
    assert len(fields[0]) == 1
    assert fields[3] == "CAN"

    # should return items within range
    geometry, fields = read(
        naturalearth_lowres, where="POP_EST >= 10000000 AND POP_EST < 100000000"
    )[2:]
    assert len(geometry) == 75
    assert min(fields[0]) >= 10000000
    assert max(fields[0]) < 100000000

    # should match no items
    with pytest.warns(UserWarning, match="does not have any features to read"):
        geometry, fields = read(naturalearth_lowres, where="iso_a3 = 'INVALID'")[2:]
        assert len(geometry) == 0


def test_read_where_invalid(naturalearth_lowres):
    with pytest.raises(ValueError, match="Invalid SQL"):
        read(naturalearth_lowres, where="invalid")


@pytest.mark.parametrize("bbox", [(1,), (1, 2), (1, 2, 3)])
def test_read_bbox_invalid(naturalearth_lowres, bbox):
    with pytest.raises(ValueError, match="Invalid bbox"):
        read(naturalearth_lowres, bbox=bbox)


def test_read_bbox(naturalearth_lowres_all_ext):
    # should return no features
    with pytest.warns(UserWarning, match="does not have any features to read"):
        geometry, fields = read(
            naturalearth_lowres_all_ext, bbox=(0, 0, 0.00001, 0.00001)
        )[2:]

    assert len(geometry) == 0

    geometry, fields = read(naturalearth_lowres_all_ext, bbox=(-85, 8, -80, 10))[2:]

    assert len(geometry) == 2
    assert np.array_equal(fields[3], ["PAN", "CRI"])


def test_read_fids(naturalearth_lowres):
    expected_fids, expected_geometry, expected_fields = read(
        naturalearth_lowres, return_fids=True
    )[1:]
    subset = [0, 10, 5]

    for fids in [subset, np.array(subset)]:
        index, geometry, fields = read(
            naturalearth_lowres, fids=subset, return_fids=True
        )[1:]

        assert len(fids) == 3
        assert len(geometry) == 3
        assert len(fields[0]) == 3

        assert np.array_equal(index, expected_fids[subset])
        assert np.array_equal(geometry, expected_geometry[subset])
        assert np.array_equal(fields[-1], expected_fields[-1][subset])


def test_return_fids(naturalearth_lowres):

    # default is to not return fids
    fids = read(naturalearth_lowres)[1]
    assert fids is None

    fids = read(naturalearth_lowres, return_fids=False)[1]
    assert fids is None

    fids = read(naturalearth_lowres, return_fids=True, skip_features=2, max_features=2)[
        1
    ]
    assert fids is not None
    assert fids.dtype == np.int64
    # Note: shapefile FIDS start at 0
    assert np.array_equal(fids, np.array([2, 3], dtype="int64"))


def test_read_fids_out_of_bounds(naturalearth_lowres):
    with pytest.raises(
        FeatureError,
        match=r"Attempt to read shape with feature id \(-1\) out of available range",
    ):
        read(naturalearth_lowres, fids=[-1])

    with pytest.raises(
        FeatureError,
        match=r"Attempt to read shape with feature id \(200\) out of available range",
    ):
        read(naturalearth_lowres, fids=[200])


def test_read_fids_unsupported_keywords(naturalearth_lowres):
    with pytest.raises(ValueError, match="cannot set both 'fids' and any of"):
        read(naturalearth_lowres, fids=[1], where="iso_a3 = 'CAN'")

    with pytest.raises(ValueError, match="cannot set both 'fids' and any of"):
        read(naturalearth_lowres, fids=[1], bbox=(-140, 20, -100, 45))

    with pytest.raises(ValueError, match="cannot set both 'fids' and any of"):
        read(naturalearth_lowres, fids=[1], skip_features=5)

    with pytest.raises(ValueError, match="cannot set both 'fids' and any of"):
        read(naturalearth_lowres, fids=[1], max_features=5)


def test_write(tmpdir, naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.shp")
    write(filename, geometry, field_data, **meta)

    assert os.path.exists(filename)
    for ext in (".dbf", ".prj"):
        assert os.path.exists(filename.replace(".shp", ext))


def test_write_gpkg(tmpdir, naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.gpkg")
    write(filename, geometry, field_data, driver="GPKG", **meta)

    assert os.path.exists(filename)


def test_write_gpkg_multiple_layers(tmpdir, naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta["geometry_type"] = "MultiPolygon"

    filename = os.path.join(str(tmpdir), "test.gpkg")
    write(filename, geometry, field_data, driver="GPKG", layer="first", **meta)

    assert os.path.exists(filename)

    assert np.array_equal(list_layers(filename), [["first", "MultiPolygon"]])

    write(filename, geometry, field_data, driver="GPKG", layer="second", **meta)

    assert np.array_equal(
        list_layers(filename), [["first", "MultiPolygon"], ["second", "MultiPolygon"]]
    )


def test_write_geojson(tmpdir, naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)

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


@pytest.mark.skipif(
    __gdal_version__ < (3, 6, 0),
    reason="OpenFileGDB write support only available for GDAL >= 3.6.0",
)
def test_write_openfilegdb(tmpdir, naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.gdb")
    write(filename, geometry, field_data, driver="OpenFileGDB", **meta)

    assert os.path.exists(filename)


@pytest.mark.parametrize("ext", DRIVERS)
def test_write_append(tmpdir, naturalearth_lowres, ext):
    if ext == ".fgb" and __gdal_version__ <= (3, 5, 0):
        pytest.skip("Append to FlatGeobuf fails for GDAL <= 3.5.0")

    if ext in (".geojsonl", ".geojsons") and __gdal_version__ < (3, 6, 0):
        pytest.skip("Append to GeoJSONSeq only available for GDAL >= 3.6.0")

    meta, _, geometry, field_data = read(naturalearth_lowres)

    # coerce output layer to MultiPolygon to avoid mixed type errors
    meta["geometry_type"] = "MultiPolygon"

    filename = os.path.join(str(tmpdir), f"test{ext}")
    write(filename, geometry, field_data, **meta)

    assert os.path.exists(filename)

    assert read_info(filename)["features"] == 177

    # write the same records again
    write(filename, geometry, field_data, append=True, **meta)

    assert read_info(filename)["features"] == 354


@pytest.mark.parametrize("driver,ext", [("GML", ".gml"), ("GeoJSONSeq", ".geojsons")])
def test_write_append_unsupported(tmpdir, naturalearth_lowres, driver, ext):
    if ext == ".geojsons" and __gdal_version__ >= (3, 6, 0):
        pytest.skip("Append to GeoJSONSeq supported for GDAL >= 3.6.0")

    meta, _, geometry, field_data = read(naturalearth_lowres)

    # GML does not support append functionality
    filename = os.path.join(str(tmpdir), f"test{ext}")
    write(filename, geometry, field_data, driver=driver, **meta)

    assert os.path.exists(filename)

    assert read_info(filename)["features"] == 177

    with pytest.raises(DataSourceError):
        write(filename, geometry, field_data, driver=driver, append=True, **meta)


@pytest.mark.skipif(
    __gdal_version__ > (3, 5, 0),
    reason="segfaults on FlatGeobuf limited to GDAL <= 3.5.0",
)
def test_write_append_prevent_gdal_segfault(tmpdir, naturalearth_lowres):
    """GDAL <= 3.5.0 segfaults when appending to FlatGeobuf; this test
    verifies that we catch that before segfault"""
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta["geometry_type"] = "MultiPolygon"

    filename = os.path.join(str(tmpdir), "test.fgb")
    write(filename, geometry, field_data, **meta)

    assert os.path.exists(filename)

    with pytest.raises(
        RuntimeError,  # match="append to FlatGeobuf is not supported for GDAL <= 3.5.0"
    ):
        write(filename, geometry, field_data, append=True, **meta)


@pytest.mark.parametrize(
    "driver",
    {
        driver
        for driver in DRIVERS.values()
        if driver not in ("ESRI Shapefile", "GPKG", "GeoJSON")
    },
)
def test_write_supported(tmpdir, naturalearth_lowres, driver):
    """Test drivers known to work that are not specifically tested above"""
    meta, _, geometry, field_data = read(naturalearth_lowres, columns=["iso_a3"])

    # note: naturalearth_lowres contains mixed polygons / multipolygons, which
    # are not supported in mixed form for all drivers.  To get around this here
    # we take the first record only.
    meta["geometry_type"] = "MultiPolygon"

    filename = tmpdir / f"test{DRIVER_EXT[driver]}"
    write(
        filename,
        geometry[:1],
        field_data=[f[:1] for f in field_data],
        driver=driver,
        **meta,
    )

    assert filename.exists()


@pytest.mark.skipif(
    __gdal_version__ >= (3, 6, 0), reason="OpenFileGDB supports write for GDAL >= 3.6.0"
)
def test_write_unsupported(tmpdir, naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.gdb")

    with pytest.raises(DataSourceError, match="does not support write functionality"):
        write(filename, geometry, field_data, driver="OpenFileGDB", **meta)


def assert_equal_result(result1, result2):
    meta1, index1, geometry1, field_data1 = result1
    meta2, index2, geometry2, field_data2 = result2

    assert np.array_equal(meta1["fields"], meta2["fields"])
    assert np.array_equal(index1, index2)
    # a plain `assert np.array_equal(geometry1, geometry2)` doesn't work because
    # the WKB values are not exactly equal, therefore parsing with pygeos to compare
    # with tolerance
    try:
        from shapely import from_wkb, equals_exact
    except ImportError:
        try:
            from pygeos import from_wkb, equals_exact
        except ImportError:
            pytest.skip("Test requires pygeos or shapely>=2")
    assert equals_exact(
        from_wkb(geometry1), from_wkb(geometry2), tolerance=0.00001
    ).all()
    assert all([np.array_equal(f1, f2) for f1, f2 in zip(field_data1, field_data2)])


@pytest.mark.parametrize("driver,ext", [("GeoJSON", "geojson"), ("GPKG", "gpkg")])
def test_read_from_bytes(tmpdir, naturalearth_lowres, driver, ext):
    meta, index, geometry, field_data = read(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), f"test.{ext}")
    write(filename, geometry, field_data, driver=driver, **meta)

    with open(filename, "rb") as f:
        buffer = f.read()

    result2 = read(buffer)
    assert_equal_result((meta, index, geometry, field_data), result2)


def test_read_from_bytes_zipped(tmpdir, naturalearth_lowres_vsi):
    path, vsi_path = naturalearth_lowres_vsi
    meta, index, geometry, field_data = read(vsi_path)

    with open(path, "rb") as f:
        buffer = f.read()

    result2 = read(buffer)
    assert_equal_result((meta, index, geometry, field_data), result2)


@pytest.mark.parametrize("driver,ext", [("GeoJSON", "geojson"), ("GPKG", "gpkg")])
def test_read_from_file_like(tmpdir, naturalearth_lowres, driver, ext):
    meta, index, geometry, field_data = read(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), f"test.{ext}")
    write(filename, geometry, field_data, driver=driver, **meta)

    with open(filename, "rb") as f:
        result2 = read(f)

    assert_equal_result((meta, index, geometry, field_data), result2)


@pytest.mark.parametrize("ext", ["gpkg", "fgb"])
def test_read_write_data_types_numeric(tmp_path, ext):
    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")] * 3, dtype=object
    )
    field_data = [
        np.array([True, False, True], dtype="bool"),
        np.array([1, 2, 3], dtype="int16"),
        np.array([1, 2, 3], dtype="int32"),
        np.array([1, 2, 3], dtype="int64"),
        np.array([1, 2, 3], dtype="float32"),
        np.array([1, 2, 3], dtype="float64"),
    ]
    fields = ["bool", "int16", "int32", "int64", "float32", "float64"]
    meta = dict(geometry_type="Point", crs="EPSG:4326", spatial_index=False)

    filename = tmp_path / f"test.{ext}"
    write(filename, geometry, field_data, fields, **meta)
    result = read(filename)[3]
    assert all([np.array_equal(f1, f2) for f1, f2 in zip(result, field_data)])
    assert all([f1.dtype == f2.dtype for f1, f2 in zip(result, field_data)])

    # other integer data types that don't roundtrip exactly
    # these are generally promoted to a larger integer type except for uint64
    for i, (dtype, result_dtype) in enumerate(
        [
            ("int8", "int16"),
            ("uint8", "int16"),
            ("uint16", "int32"),
            ("uint32", "int64"),
            ("uint64", "int64"),
        ]
    ):
        field_data = [np.array([1, 2, 3], dtype=dtype)]
        filename = tmp_path / f"test{i}.{ext}"
        write(filename, geometry, field_data, ["col"], **meta)
        result = read(filename)[3][0]
        assert np.array_equal(result, np.array([1, 2, 3]))
        assert result.dtype == result_dtype


def test_read_write_datetime(tmp_path):
    field_data = [
        np.array(["2005-02-01", "2005-02-02"], dtype="datetime64[D]"),
        np.array(["2001-01-01T12:00", "2002-02-03T13:56:03"], dtype="datetime64[s]"),
        np.array(
            ["2001-01-01T12:00", "2002-02-03T13:56:03.072"], dtype="datetime64[ms]"
        ),
        np.array(
            ["2001-01-01T12:00", "2002-02-03T13:56:03.072"], dtype="datetime64[ns]"
        ),
        np.array(
            ["2001-01-01T12:00", "2002-02-03T13:56:03.072123456"],
            dtype="datetime64[ns]",
        ),
        # Remark: a None value is automatically converted to np.datetime64("NaT")
        np.array([np.datetime64("NaT"), None], dtype="datetime64[ms]"),
    ]
    fields = [
        "datetime64_d",
        "datetime64_s",
        "datetime64_ms",
        "datetime64_ns",
        "datetime64_precise_ns",
        "datetime64_ms_nat",
    ]

    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")] * 2, dtype=object
    )
    meta = dict(geometry_type="Point", crs="EPSG:4326", spatial_index=False)

    filename = tmp_path / "test.gpkg"
    write(filename, geometry, field_data, fields, **meta)
    result = read(filename)[3]
    for idx, field in enumerate(fields):
        if field == "datetime64_precise_ns":
            # gdal rounds datetimes to ms
            assert np.array_equal(result[idx], field_data[idx].astype("datetime64[ms]"))
        else:
            assert np.array_equal(result[idx], field_data[idx], equal_nan=True)


def test_read_data_types_numeric_with_null(test_gpkg_nulls):
    fields = read(test_gpkg_nulls)[3]

    for i, field in enumerate(fields):
        # last value should be np.nan
        assert np.isnan(field[-1])

        # all integer fields should be cast to float64; float32 should be preserved
        if i == 9:
            assert field.dtype == "float32"
        else:
            assert field.dtype == "float64"


def test_read_unsupported_types(test_ogr_types_list):
    fields = read(test_ogr_types_list)[3]
    # list field gets skipped, only integer field is read
    assert len(fields) == 1

    fields = read(test_ogr_types_list, columns=["int64"])[3]
    assert len(fields) == 1


def test_read_datetime_millisecond(test_datetime):
    field = read(test_datetime)[3][0]
    assert field.dtype == "datetime64[ms]"
    assert field[0] == np.datetime64("2020-01-01 09:00:00.123")
    assert field[1] == np.datetime64("2020-01-01 10:00:00.000")


@pytest.mark.parametrize("ext", ["gpkg", "geojson"])
def test_read_write_null_geometry(tmp_path, ext):
    # Point(0, 0), null
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000"), None],
        dtype=object,
    )
    field_data = [np.array([1, 2], dtype="int32")]
    fields = ["col"]
    meta = dict(geometry_type="Point", crs="EPSG:4326")
    if ext == "gpkg":
        meta["spatial_index"] = False

    filename = tmp_path / f"test.{ext}"
    write(filename, geometry, field_data, fields, **meta)
    result_geometry, result_fields = read(filename)[2:]
    assert np.array_equal(result_geometry, geometry)
    assert np.array_equal(result_fields[0], field_data[0])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_write_float_nan_null(tmp_path, dtype):
    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")] * 2,
        dtype=object,
    )
    field_data = [np.array([1.5, np.nan], dtype=dtype)]
    fields = ["col"]
    meta = dict(geometry_type="Point", crs="EPSG:4326")
    fname = tmp_path / "test.geojson"

    # default nan_as_null=True
    write(fname, geometry, field_data, fields, **meta)
    with open(str(fname), "r") as f:
        content = f.read()
    assert '{ "col": null }' in content

    # set to False
    # by default, GDAL will skip the property for GeoJSON if the value is NaN
    write(fname, geometry, field_data, fields, **meta, nan_as_null=False)
    with open(str(fname), "r") as f:
        content = f.read()
    assert '"properties": { }' in content

    # but can instruct GDAL to write NaN to json
    write(
        fname,
        geometry,
        field_data,
        fields,
        **meta,
        nan_as_null=False,
        WRITE_NON_FINITE_VALUES="YES",
    )
    with open(str(fname), "r") as f:
        content = f.read()
    assert '{ "col": NaN }' in content


@pytest.mark.skipif("Arrow" not in list_drivers(), reason="GDAL not built with Arrow")
def test_write_float_nan_null_arrow(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.feather

    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")] * 2,
        dtype=object,
    )
    field_data = [np.array([1.5, np.nan], dtype="float64")]
    fields = ["col"]
    meta = dict(geometry_type="Point", crs="EPSG:4326")
    fname = tmp_path / "test.arrow"

    # default nan_as_null=True
    write(fname, geometry, field_data, fields, driver="Arrow", **meta)
    table = pyarrow.feather.read_table(fname)
    assert table["col"].is_null().to_pylist() == [False, True]

    # set to False
    write(
        fname, geometry, field_data, fields, driver="Arrow", nan_as_null=False, **meta
    )
    table = pyarrow.feather.read_table(fname)
    assert table["col"].is_null().to_pylist() == [False, False]
    pc = pytest.importorskip("pyarrow.compute")
    assert pc.is_nan(table["col"]).to_pylist() == [False, True]


@pytest.mark.parametrize("ext", ["fgb", "gpkg", "geojson"])
@pytest.mark.parametrize(
    "read_encoding,write_encoding",
    [
        pytest.param(
            None,
            None,
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="must specify write encoding on Windows"
            ),
        ),
        pytest.param(
            "UTF-8",
            None,
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="must specify write encoding on Windows"
            ),
        ),
        (None, "UTF-8"),
        ("UTF-8", "UTF-8"),
    ],
)
def test_encoding_io(tmp_path, ext, read_encoding, write_encoding):
    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")], dtype=object
    )
    arabic = "العربية"
    cree = "ᓀᐦᐃᔭᐍᐏᐣ"
    mandarin = "中文"
    field_data = [
        np.array([arabic], dtype=object),
        np.array([cree], dtype=object),
        np.array([mandarin], dtype=object),
    ]
    fields = [arabic, cree, mandarin]
    meta = dict(geometry_type="Point", crs="EPSG:4326", encoding=write_encoding)

    filename = tmp_path / f"test.{ext}"
    write(filename, geometry, field_data, fields, **meta)

    actual_meta, _, _, actual_field_data = read(filename, encoding=read_encoding)
    assert np.array_equal(fields, actual_meta["fields"])
    assert np.array_equal(field_data, actual_field_data)
    assert np.array_equal(fields, read_info(filename, encoding=read_encoding)["fields"])


@pytest.mark.parametrize(
    "read_encoding,write_encoding",
    [
        pytest.param(
            None,
            None,
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="must specify write encoding on Windows"
            ),
        ),
        pytest.param(
            "UTF-8",
            None,
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="must specify write encoding on Windows"
            ),
        ),
        (None, "UTF-8"),
        ("UTF-8", "UTF-8"),
    ],
)
def test_encoding_io_shapefile(tmp_path, read_encoding, write_encoding):
    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")], dtype=object
    )
    arabic = "العربية"
    cree = "ᓀᐦᐃᔭᐍᐏᐣ"
    mandarin = "中文"
    field_data = [
        np.array([arabic], dtype=object),
        np.array([cree], dtype=object),
        np.array([mandarin], dtype=object),
    ]

    # Field names are longer than 10 bytes and get truncated badly (not at UTF-8
    # character level)  by GDAL when output to shapefile, so we have to truncate
    # before writing
    fields = [arabic[:5], cree[:3], mandarin]
    meta = dict(geometry_type="Point", crs="EPSG:4326", encoding="UTF-8")

    filename = tmp_path / "test.shp"
    # NOTE: GDAL automatically creates a cpg file with the encoding name, which
    # means that if we read this without specifying the encoding it uses the
    # correct one
    write(filename, geometry, field_data, fields, **meta)

    actual_meta, _, _, actual_field_data = read(filename, encoding=read_encoding)
    assert np.array_equal(fields, actual_meta["fields"])
    assert np.array_equal(field_data, actual_field_data)
    assert np.array_equal(fields, read_info(filename, encoding=read_encoding)["fields"])

    # verify that if cpg file is not present, that user-provided encoding is used,
    # otherwise it defaults to ISO-8859-1
    if read_encoding is not None:
        os.unlink(str(filename).replace(".shp", ".cpg"))
        actual_meta, _, _, actual_field_data = read(filename, encoding=read_encoding)
        assert np.array_equal(fields, actual_meta["fields"])
        assert np.array_equal(field_data, actual_field_data)
        assert np.array_equal(
            fields, read_info(filename, encoding=read_encoding)["fields"]
        )
