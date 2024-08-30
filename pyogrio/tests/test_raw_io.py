import contextlib
import ctypes
import json
import sys
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from numpy import array_equal

import pyogrio
from pyogrio import (
    __gdal_version__,
    get_gdal_config_option,
    list_drivers,
    list_layers,
    read_info,
    set_gdal_config_options,
)
from pyogrio._compat import HAS_PYARROW, HAS_SHAPELY
from pyogrio.errors import DataLayerError, DataSourceError, FeatureError
from pyogrio.raw import open_arrow, read, write
from pyogrio.tests.conftest import (
    DRIVER_EXT,
    DRIVERS,
    prepare_testfile,
    requires_arrow_api,
    requires_pyarrow_api,
    requires_shapely,
)

import pytest

try:
    import shapely
except ImportError:
    pass


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


def test_read_arrow_unspecified_layer_warning(data_dir):
    """Reading a multi-layer file without specifying a layer gives a warning."""
    with pytest.warns(UserWarning, match="More than one layer found "):
        read(data_dir / "sample.osm.pbf")


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


@requires_shapely
def test_read_no_geometry__mask(naturalearth_lowres):
    geometry, fields = read(
        naturalearth_lowres,
        read_geometry=False,
        mask=shapely.Point(-105, 55),
    )[2:]

    assert np.array_equal(fields[3], ["CAN"])
    assert geometry is None


def test_read_no_geometry__bbox(naturalearth_lowres):
    geometry, fields = read(
        naturalearth_lowres,
        read_geometry=False,
        bbox=(-109.0, 55.0, -109.0, 55.0),
    )[2:]

    assert np.array_equal(fields[3], ["CAN"])
    assert geometry is None


def test_read_no_geometry_no_columns_no_fids(naturalearth_lowres):
    with pytest.raises(
        ValueError,
        match=(
            "at least one of read_geometry or return_fids must be True or columns must "
            "be None or non-empty"
        ),
    ):
        _ = read(
            naturalearth_lowres, columns=[], read_geometry=False, return_fids=False
        )


def test_read_columns(naturalearth_lowres):
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


@pytest.mark.parametrize("skip_features", [10, 200])
def test_read_skip_features(naturalearth_lowres_all_ext, skip_features):
    expected_geometry, expected_fields = read(naturalearth_lowres_all_ext)[2:]
    geometry, fields = read(naturalearth_lowres_all_ext, skip_features=skip_features)[
        2:
    ]

    # skipping more features than available in layer returns empty arrays
    expected_count = max(len(expected_geometry) - skip_features, 0)

    assert len(geometry) == expected_count
    assert len(fields[0]) == expected_count

    assert np.array_equal(geometry, expected_geometry[skip_features:])
    # Last field has more variable data
    assert np.array_equal(fields[-1], expected_fields[-1][skip_features:])


def test_read_negative_skip_features(naturalearth_lowres):
    with pytest.raises(ValueError, match="'skip_features' must be >= 0"):
        read(naturalearth_lowres, skip_features=-1)


def test_read_max_features(naturalearth_lowres):
    expected_geometry, expected_fields = read(naturalearth_lowres)[2:]
    geometry, fields = read(naturalearth_lowres, max_features=2)[2:]

    assert len(geometry) == 2
    assert len(fields[0]) == 2

    assert np.array_equal(geometry, expected_geometry[:2])
    assert np.array_equal(fields[-1], expected_fields[-1][:2])


def test_read_negative_max_features(naturalearth_lowres):
    with pytest.raises(ValueError, match="'max_features' must be >= 0"):
        read(naturalearth_lowres, max_features=-1)


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
    geometry, fields = read(naturalearth_lowres_all_ext, bbox=(0, 0, 0.00001, 0.00001))[
        2:
    ]

    assert len(geometry) == 0

    geometry, fields = read(naturalearth_lowres_all_ext, bbox=(-85, 8, -80, 10))[2:]

    assert len(geometry) == 2
    assert np.array_equal(fields[3], ["PAN", "CRI"])


def test_read_bbox_sql(naturalearth_lowres_all_ext):
    fields = read(
        naturalearth_lowres_all_ext,
        bbox=(-180, 50, -100, 90),
        sql="SELECT * from naturalearth_lowres where iso_a3 not in ('USA', 'RUS')",
    )[3]
    assert len(fields[3]) == 1
    assert np.array_equal(fields[3], ["CAN"])


def test_read_bbox_where(naturalearth_lowres_all_ext):
    fields = read(
        naturalearth_lowres_all_ext,
        bbox=(-180, 50, -100, 90),
        where="iso_a3 not in ('USA', 'RUS')",
    )[3]
    assert len(fields[3]) == 1
    assert np.array_equal(fields[3], ["CAN"])


@requires_shapely
@pytest.mark.parametrize(
    "mask",
    [
        {"type": "Point", "coordinates": [0, 0]},
        '{"type": "Point", "coordinates": [0, 0]}',
        "invalid",
    ],
)
def test_read_mask_invalid(naturalearth_lowres, mask):
    with pytest.raises(ValueError, match="'mask' parameter must be a Shapely geometry"):
        read(naturalearth_lowres, mask=mask)


@requires_shapely
def test_read_bbox_mask_invalid(naturalearth_lowres):
    with pytest.raises(ValueError, match="cannot set both 'bbox' and 'mask'"):
        read(naturalearth_lowres, bbox=(-85, 8, -80, 10), mask=shapely.Point(-105, 55))


@requires_shapely
@pytest.mark.parametrize(
    "mask,expected",
    [
        ("POINT (-105 55)", ["CAN"]),
        ("POLYGON ((-80 8, -80 10, -85 10, -85 8, -80 8))", ["PAN", "CRI"]),
        (
            """POLYGON ((
                6.101929 50.97085,
                5.773002 50.906611,
                5.593156 50.642649,
                6.059271 50.686052,
                6.374064 50.851481,
                6.101929 50.97085
            ))""",
            ["DEU", "BEL", "NLD"],
        ),
        (
            """GEOMETRYCOLLECTION (
                POINT (-7.7 53),
                POLYGON ((-80 8, -80 10, -85 10, -85 8, -80 8))
            )""",
            ["PAN", "CRI", "IRL"],
        ),
    ],
)
def test_read_mask(naturalearth_lowres_all_ext, mask, expected):
    mask = shapely.from_wkt(mask)

    geometry, fields = read(naturalearth_lowres_all_ext, mask=mask)[2:]

    assert np.array_equal(fields[3], expected)
    assert len(geometry) == len(expected)


@requires_shapely
def test_read_mask_sql(naturalearth_lowres_all_ext):
    fields = read(
        naturalearth_lowres_all_ext,
        mask=shapely.box(-180, 50, -100, 90),
        sql="SELECT * from naturalearth_lowres where iso_a3 not in ('USA', 'RUS')",
    )[3]
    assert len(fields[3]) == 1
    assert np.array_equal(fields[3], ["CAN"])


@requires_shapely
def test_read_mask_where(naturalearth_lowres_all_ext):
    fields = read(
        naturalearth_lowres_all_ext,
        mask=shapely.box(-180, 50, -100, 90),
        where="iso_a3 not in ('USA', 'RUS')",
    )[3]
    assert len(fields[3]) == 1
    assert np.array_equal(fields[3], ["CAN"])


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

    with pytest.raises(ValueError, match="cannot set both 'fids' and any of"):
        read(naturalearth_lowres, fids=[1], bbox=(0, 0, 0.0001, 0.0001))

    if HAS_SHAPELY:
        with pytest.raises(ValueError, match="cannot set both 'fids' and any of"):
            read(naturalearth_lowres, fids=[1], mask=shapely.Point(0, 0))


def test_read_return_fids(naturalearth_lowres):
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


def test_read_return_only_fids(naturalearth_lowres):
    _, fids, geometry, field_data = read(
        naturalearth_lowres, columns=[], read_geometry=False, return_fids=True
    )
    assert fids is not None
    assert len(fids) == 177
    assert geometry is None
    assert len(field_data) == 0


@pytest.mark.parametrize("encoding", [None, "ISO-8859-1"])
def test_write_shp(tmp_path, naturalearth_lowres, encoding):
    meta, _, geometry, field_data = read(naturalearth_lowres)

    filename = tmp_path / "test.shp"
    meta["encoding"] = encoding
    write(filename, geometry, field_data, **meta)

    assert filename.exists()
    for ext in (".dbf", ".prj"):
        assert filename.with_suffix(ext).exists()

    # We write shapefiles in UTF-8 by default on all platforms
    expected_encoding = encoding if encoding is not None else "UTF-8"
    with open(filename.with_suffix(".cpg")) as cpg_file:
        result_encoding = cpg_file.read()
        assert result_encoding == expected_encoding


def test_write_gpkg(tmp_path, naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta.update({"geometry_type": "MultiPolygon"})

    filename = tmp_path / "test.gpkg"
    write(filename, geometry, field_data, driver="GPKG", **meta)

    assert filename.exists()


def test_write_gpkg_multiple_layers(tmp_path, naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta["geometry_type"] = "MultiPolygon"

    filename = tmp_path / "test.gpkg"
    write(filename, geometry, field_data, driver="GPKG", layer="first", **meta)

    assert filename.exists()

    assert np.array_equal(list_layers(filename), [["first", "MultiPolygon"]])

    write(filename, geometry, field_data, driver="GPKG", layer="second", **meta)

    assert np.array_equal(
        list_layers(filename), [["first", "MultiPolygon"], ["second", "MultiPolygon"]]
    )


def test_write_geojson(tmp_path, naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)

    filename = tmp_path / "test.json"
    write(filename, geometry, field_data, driver="GeoJSON", **meta)

    assert filename.exists()

    data = json.loads(open(filename).read())

    assert data["type"] == "FeatureCollection"
    assert data["name"] == "test"
    assert "crs" in data
    assert len(data["features"]) == len(geometry)
    assert not len(
        set(meta["fields"]).difference(data["features"][0]["properties"].keys())
    )


def test_write_no_fields(tmp_path, naturalearth_lowres):
    """Test writing file with no fields/attribute columns."""
    # Prepare test data
    meta, _, geometry, field_data = read(naturalearth_lowres)
    field_data = None
    meta["fields"] = None
    # naturalearth_lowres actually contains MultiPolygons. A shapefile doesn't make the
    # distinction, so the metadata just reports Polygon. GPKG does, so override here to
    # avoid GDAL warnings.
    meta["geometry_type"] = "MultiPolygon"

    # Test
    filename = tmp_path / "test.gpkg"
    write(filename, geometry, field_data, driver="GPKG", **meta)

    # Check result
    assert filename.exists()
    meta, _, geometry, fields = read(filename)

    assert meta["crs"] == "EPSG:4326"
    assert meta["geometry_type"] == "MultiPolygon"
    assert meta["encoding"] == "UTF-8"
    assert meta["fields"].shape == (0,)
    assert len(fields) == 0
    assert len(geometry) == 177

    # quick test that WKB is a Polygon type
    assert geometry[0][:6] == b"\x01\x06\x00\x00\x00\x03"


def test_write_no_geom(tmp_path, naturalearth_lowres):
    """Test writing file with no geometry column."""
    # Prepare test data
    meta, _, geometry, field_data = read(naturalearth_lowres)
    geometry = None
    meta["geometry_type"] = None

    # Test
    filename = tmp_path / "test.gpkg"
    write(filename, geometry, field_data, driver="GPKG", **meta)

    # Check result
    assert filename.exists()
    meta, _, geometry, fields = read(filename)

    assert meta["crs"] is None
    assert meta["geometry_type"] is None
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
    assert len(fields[0]) == 177


def test_write_no_geom_data(tmp_path, naturalearth_lowres):
    """Test writing file with no geometry data passed but a geometry_type specified.

    In this case the geometry_type is ignored, so a file without geometry column is
    written.
    """
    # Prepare test data
    meta, _, geometry, field_data = read(naturalearth_lowres)
    # If geometry data is set to None, meta["geometry_type"] is ignored and so no
    # geometry column will be created.
    geometry = None

    # Test
    filename = tmp_path / "test.gpkg"
    write(filename, geometry, field_data, driver="GPKG", **meta)

    # Check result
    assert filename.exists()
    result_meta, _, result_geometry, result_field_data = read(filename)

    assert result_meta["crs"] is None
    assert result_meta["geometry_type"] is None
    assert result_meta["encoding"] == "UTF-8"
    assert result_meta["fields"].shape == (5,)

    assert result_meta["fields"].tolist() == [
        "pop_est",
        "continent",
        "name",
        "iso_a3",
        "gdp_md_est",
    ]

    assert len(result_field_data) == 5
    assert len(result_field_data[0]) == 177
    assert result_geometry is None


def test_write_no_geom_no_fields():
    """Test writing file with no geometry column nor fields -> error."""
    with pytest.raises(
        ValueError,
        match="You must provide at least a geometry column or a field",
    ):
        write("test.gpkg", geometry=None, field_data=None, fields=None)


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
    # Point(0, 0)
    expected_geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")] * 3, dtype=object
    )
    expected_field_data = [
        np.array([True, False, True], dtype="bool"),
        np.array([1, 2, 3], dtype="int16"),
        np.array([1, 2, 3], dtype="int32"),
        np.array([1, 2, 3], dtype="int64"),
        np.array([1, 2, 3], dtype="float32"),
        np.array([1, 2, 3], dtype="float64"),
    ]
    expected_fields = ["bool", "int16", "int32", "int64", "float32", "float64"]
    expected_meta = {
        "geometry_type": "Point",
        "crs": "EPSG:4326",
        "fields": expected_fields,
    }

    filename = tmp_path / "test.gdb"

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
        write(
            filename,
            expected_geometry,
            expected_field_data,
            driver="OpenFileGDB",
            **expected_meta,
            **write_params,
        )

    meta, _, geometry, field_data = read(filename)

    if not write_int64:
        expected_field_data[3] = expected_field_data[3].astype("float64")

    # bool types are converted to int32
    expected_field_data[0] = expected_field_data[0].astype("int32")

    assert meta["crs"] == expected_meta["crs"]
    assert np.array_equal(meta["fields"], expected_meta["fields"])

    assert np.array_equal(geometry, expected_geometry)
    for i in range(len(expected_field_data)):
        assert field_data[i].dtype == expected_field_data[i].dtype
        assert np.array_equal(field_data[i], expected_field_data[i])


@pytest.mark.parametrize("ext", DRIVERS)
def test_write_append(tmp_path, naturalearth_lowres, ext):
    if ext == ".fgb" and __gdal_version__ <= (3, 5, 0):
        pytest.skip("Append to FlatGeobuf fails for GDAL <= 3.5.0")

    if ext in (".geojsonl", ".geojsons") and __gdal_version__ < (3, 6, 0):
        pytest.skip("Append to GeoJSONSeq only available for GDAL >= 3.6.0")

    meta, _, geometry, field_data = read(naturalearth_lowres)

    # coerce output layer to MultiPolygon to avoid mixed type errors
    meta["geometry_type"] = "MultiPolygon"

    filename = tmp_path / f"test{ext}"
    write(filename, geometry, field_data, **meta)

    assert filename.exists()

    assert read_info(filename)["features"] == 177

    # write the same records again
    write(filename, geometry, field_data, append=True, **meta)

    assert read_info(filename)["features"] == 354


@pytest.mark.parametrize("driver,ext", [("GML", ".gml"), ("GeoJSONSeq", ".geojsons")])
def test_write_append_unsupported(tmp_path, naturalearth_lowres, driver, ext):
    if ext == ".geojsons" and __gdal_version__ >= (3, 6, 0):
        pytest.skip("Append to GeoJSONSeq supported for GDAL >= 3.6.0")

    meta, _, geometry, field_data = read(naturalearth_lowres)

    # GML does not support append functionality
    filename = tmp_path / f"test{ext}"
    write(filename, geometry, field_data, driver=driver, **meta)

    assert filename.exists()

    assert read_info(filename, force_feature_count=True)["features"] == 177

    with pytest.raises(DataSourceError):
        write(filename, geometry, field_data, driver=driver, append=True, **meta)


@pytest.mark.skipif(
    __gdal_version__ > (3, 5, 0),
    reason="segfaults on FlatGeobuf limited to GDAL <= 3.5.0",
)
def test_write_append_prevent_gdal_segfault(tmp_path, naturalearth_lowres):
    """GDAL <= 3.5.0 segfaults when appending to FlatGeobuf; this test
    verifies that we catch that before segfault"""
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta["geometry_type"] = "MultiPolygon"

    filename = tmp_path / "test.fgb"
    write(filename, geometry, field_data, **meta)

    assert filename.exists()

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
def test_write_supported(tmp_path, naturalearth_lowres, driver):
    """Test drivers known to work that are not specifically tested above"""
    meta, _, geometry, field_data = read(naturalearth_lowres, columns=["iso_a3"])

    # note: naturalearth_lowres contains mixed polygons / multipolygons, which
    # are not supported in mixed form for all drivers.  To get around this here
    # we take the first record only.
    meta["geometry_type"] = "MultiPolygon"

    filename = tmp_path / f"test{DRIVER_EXT[driver]}"
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
def test_write_unsupported(tmp_path, naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)

    filename = tmp_path / "test.gdb"

    with pytest.raises(DataSourceError, match="does not support write functionality"):
        write(filename, geometry, field_data, driver="OpenFileGDB", **meta)


def test_write_gdalclose_error(naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)

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
        write(filename, geometry, field_data, **meta)


def assert_equal_result(result1, result2):
    meta1, index1, geometry1, field_data1 = result1
    meta2, index2, geometry2, field_data2 = result2

    assert np.array_equal(meta1["fields"], meta2["fields"])
    assert np.array_equal(index1, index2)
    assert all(np.array_equal(f1, f2) for f1, f2 in zip(field_data1, field_data2))

    if HAS_SHAPELY:
        # a plain `assert np.array_equal(geometry1, geometry2)` doesn't work
        # because the WKB values are not exactly equal, therefore parsing with
        # shapely to compare with tolerance
        assert shapely.equals_exact(
            shapely.from_wkb(geometry1), shapely.from_wkb(geometry2), tolerance=0.00001
        ).all()


@pytest.mark.filterwarnings("ignore:File /vsimem:RuntimeWarning")  # TODO
@pytest.mark.parametrize("driver,ext", [("GeoJSON", "geojson"), ("GPKG", "gpkg")])
def test_read_from_bytes(tmp_path, naturalearth_lowres, driver, ext):
    meta, index, geometry, field_data = read(naturalearth_lowres)
    meta.update({"geometry_type": "Unknown"})
    filename = tmp_path / f"test.{ext}"
    write(filename, geometry, field_data, driver=driver, **meta)

    with open(filename, "rb") as f:
        buffer = f.read()

    result2 = read(buffer)
    assert_equal_result((meta, index, geometry, field_data), result2)


def test_read_from_bytes_zipped(naturalearth_lowres_vsi):
    path, vsi_path = naturalearth_lowres_vsi
    meta, index, geometry, field_data = read(vsi_path)

    with open(path, "rb") as f:
        buffer = f.read()

    result2 = read(buffer)
    assert_equal_result((meta, index, geometry, field_data), result2)


@pytest.mark.filterwarnings("ignore:File /vsimem:RuntimeWarning")  # TODO
@pytest.mark.parametrize("driver,ext", [("GeoJSON", "geojson"), ("GPKG", "gpkg")])
def test_read_from_file_like(tmp_path, naturalearth_lowres, driver, ext):
    meta, index, geometry, field_data = read(naturalearth_lowres)
    meta.update({"geometry_type": "Unknown"})
    filename = tmp_path / f"test.{ext}"
    write(filename, geometry, field_data, driver=driver, **meta)

    with open(filename, "rb") as f:
        result2 = read(f)

    assert_equal_result((meta, index, geometry, field_data), result2)


def test_read_from_nonseekable_bytes(nonseekable_bytes):
    meta, _, geometry, _ = read(nonseekable_bytes)
    assert meta["fields"].shape == (0,)
    assert len(geometry) == 1


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
    meta = {"geometry_type": "Point", "crs": "EPSG:4326", "spatial_index": False}

    filename = tmp_path / f"test.{ext}"
    write(filename, geometry, field_data, fields, **meta)
    result = read(filename)[3]
    assert all(np.array_equal(f1, f2) for f1, f2 in zip(result, field_data))
    assert all(f1.dtype == f2.dtype for f1, f2 in zip(result, field_data))

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
    meta = {"geometry_type": "Point", "crs": "EPSG:4326", "spatial_index": False}

    filename = tmp_path / "test.gpkg"
    write(filename, geometry, field_data, fields, **meta)
    result = read(filename)[3]
    for idx, field in enumerate(fields):
        if field == "datetime64_precise_ns":
            # gdal rounds datetimes to ms
            assert np.array_equal(result[idx], field_data[idx].astype("datetime64[ms]"))
        else:
            assert np.array_equal(result[idx], field_data[idx], equal_nan=True)


@pytest.mark.parametrize("ext", ["gpkg", "fgb"])
def test_read_write_int64_large(tmp_path, ext):
    # Test if value > max int32 is correctly written and read.
    # Test introduced to validate https://github.com/geopandas/pyogrio/issues/259
    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")] * 3, dtype=object
    )
    field_data = [np.array([1, 2192502720, -5], dtype="int64")]
    fields = ["overflow_int64"]
    meta = {"geometry_type": "Point", "crs": "EPSG:4326", "spatial_index": False}

    filename = tmp_path / f"test.{ext}"
    write(filename, geometry, field_data, fields, **meta)
    result = read(filename)[3]
    assert np.array_equal(result, field_data)
    assert result[0].dtype == field_data[0].dtype


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


def test_read_unsupported_types(list_field_values_file):
    fields = read(list_field_values_file)[3]
    # list field gets skipped, only integer field is read
    assert len(fields) == 1

    fields = read(list_field_values_file, columns=["int64"])[3]
    assert len(fields) == 1


def test_read_datetime_millisecond(datetime_file):
    field = read(datetime_file)[3][0]
    assert field.dtype == "datetime64[ms]"
    assert field[0] == np.datetime64("2020-01-01 09:00:00.123")
    assert field[1] == np.datetime64("2020-01-01 10:00:00.000")


def test_read_unsupported_ext(tmp_path):
    test_unsupported_path = tmp_path / "test.unsupported"
    with open(test_unsupported_path, "w") as file:
        file.write("column1,column2\n")
        file.write("data1,data2")

    with pytest.raises(
        DataSourceError, match=".* by prefixing the file path with '<DRIVER>:'.*"
    ):
        read(test_unsupported_path)


def test_read_unsupported_ext_with_prefix(tmp_path):
    test_unsupported_path = tmp_path / "test.unsupported"
    with open(test_unsupported_path, "w") as file:
        file.write("column1,column2\n")
        file.write("data1,data2")

    _, _, _, field_data = read(f"CSV:{test_unsupported_path}")
    assert len(field_data) == 2
    assert field_data[0] == "data1"


def test_read_datetime_as_string(datetime_tz_file):
    field = read(datetime_tz_file)[3][0]
    assert field.dtype == "datetime64[ms]"
    # timezone is ignored in numpy layer
    assert field[0] == np.datetime64("2020-01-01 09:00:00.123")
    assert field[1] == np.datetime64("2020-01-01 10:00:00.000")

    field = read(datetime_tz_file, datetime_as_string=True)[3][0]
    assert field.dtype == "object"
    # GDAL doesn't return strings in ISO format (yet)
    assert field[0] == "2020/01/01 09:00:00.123-05"
    assert field[1] == "2020/01/01 10:00:00-05"


@pytest.mark.parametrize("ext", ["gpkg", "geojson"])
def test_read_write_null_geometry(tmp_path, ext):
    # Point(0, 0), null
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000"), None],
        dtype=object,
    )
    field_data = [np.array([1, 2], dtype="int32")]
    fields = ["col"]
    meta = {"geometry_type": "Point", "crs": "EPSG:4326"}
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
    meta = {"geometry_type": "Point", "crs": "EPSG:4326"}
    filename = tmp_path / "test.geojson"

    # default nan_as_null=True
    write(filename, geometry, field_data, fields, **meta)
    with open(filename) as f:
        content = f.read()
    assert '{ "col": null }' in content

    # set to False
    # by default, GDAL will skip the property for GeoJSON if the value is NaN
    if dtype == "float32":
        ctx = pytest.warns(RuntimeWarning, match="NaN of Infinity value found. Skipped")
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        write(filename, geometry, field_data, fields, **meta, nan_as_null=False)
    with open(filename) as f:
        content = f.read()
    assert '"properties": { }' in content

    # but can instruct GDAL to write NaN to json
    write(
        filename,
        geometry,
        field_data,
        fields,
        **meta,
        nan_as_null=False,
        WRITE_NON_FINITE_VALUES="YES",
    )
    with open(filename) as f:
        content = f.read()
    assert '{ "col": NaN }' in content


@requires_pyarrow_api
@pytest.mark.skipif(
    "Arrow" not in list_drivers(), reason="Arrow driver is not available"
)
def test_write_float_nan_null_arrow(tmp_path):
    import pyarrow.feather

    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")] * 2,
        dtype=object,
    )
    field_data = [np.array([1.5, np.nan], dtype="float64")]
    fields = ["col"]
    meta = {"geometry_type": "Point", "crs": "EPSG:4326"}
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


@pytest.mark.filterwarnings("ignore:File /vsimem:RuntimeWarning")
@pytest.mark.parametrize("driver", ["GeoJSON", "GPKG"])
def test_write_memory(naturalearth_lowres, driver):
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta.update({"geometry_type": "MultiPolygon"})

    buffer = BytesIO()
    write(buffer, geometry, field_data, driver=driver, layer="test", **meta)

    assert len(buffer.getbuffer()) > 0
    assert list_layers(buffer)[0][0] == "test"

    actual_meta, _, actual_geometry, actual_field_data = read(buffer)

    assert np.array_equal(actual_meta["fields"], meta["fields"])
    assert np.array_equal(actual_field_data, field_data)
    assert len(actual_geometry) == len(geometry)


def test_write_memory_driver_required(naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)

    buffer = BytesIO()
    with pytest.raises(
        ValueError,
        match="driver must be provided to write to in-memory file",
    ):
        write(buffer, geometry, field_data, driver=None, layer="test", **meta)


@pytest.mark.parametrize("driver", ["ESRI Shapefile", "OpenFileGDB"])
def test_write_memory_unsupported_driver(naturalearth_lowres, driver):
    if driver == "OpenFileGDB" and __gdal_version__ < (3, 6, 0):
        pytest.skip("OpenFileGDB write support only available for GDAL >= 3.6.0")

    meta, _, geometry, field_data = read(naturalearth_lowres)

    buffer = BytesIO()

    with pytest.raises(
        ValueError, match=f"writing to in-memory file is not supported for {driver}"
    ):
        write(
            buffer,
            geometry,
            field_data,
            driver=driver,
            layer="test",
            append=True,
            **meta,
        )


@pytest.mark.parametrize("driver", ["GeoJSON", "GPKG"])
def test_write_memory_append_unsupported(naturalearth_lowres, driver):
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta.update({"geometry_type": "MultiPolygon"})

    buffer = BytesIO()

    with pytest.raises(
        NotImplementedError, match="append is not supported for in-memory files"
    ):
        write(
            buffer,
            geometry,
            field_data,
            driver=driver,
            layer="test",
            append=True,
            **meta,
        )


def test_write_memory_existing_unsupported(naturalearth_lowres):
    meta, _, geometry, field_data = read(naturalearth_lowres)

    buffer = BytesIO(b"0000")
    with pytest.raises(
        NotImplementedError,
        match="writing to existing in-memory object is not supported",
    ):
        write(buffer, geometry, field_data, driver="GeoJSON", layer="test", **meta)


def test_write_open_file_handle(tmp_path, naturalearth_lowres):
    """Verify that writing to an open file handle is not currently supported"""

    meta, _, geometry, field_data = read(naturalearth_lowres)

    # verify it fails for regular file handle
    with pytest.raises(
        NotImplementedError, match="writing to an open file handle is not yet supported"
    ):
        with open(tmp_path / "test.geojson", "wb") as f:
            write(f, geometry, field_data, driver="GeoJSON", layer="test", **meta)

    # verify it fails for ZipFile
    with pytest.raises(
        NotImplementedError, match="writing to an open file handle is not yet supported"
    ):
        with ZipFile(tmp_path / "test.geojson.zip", "w") as z:
            with z.open("test.geojson", "w") as f:
                write(f, geometry, field_data, driver="GeoJSON", layer="test", **meta)


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
    meta = {"geometry_type": "Point", "crs": "EPSG:4326", "encoding": write_encoding}

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
    meta = {"geometry_type": "Point", "crs": "EPSG:4326", "encoding": "UTF-8"}

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
        filename.with_suffix(".cpg").unlink()
        actual_meta, _, _, actual_field_data = read(filename, encoding=read_encoding)
        assert np.array_equal(fields, actual_meta["fields"])
        assert np.array_equal(field_data, actual_field_data)
        assert np.array_equal(
            fields, read_info(filename, encoding=read_encoding)["fields"]
        )


@pytest.mark.parametrize("ext", ["gpkg", "geojson"])
def test_non_utf8_encoding_io(tmp_path, ext, encoded_text):
    """Verify that we write non-UTF data to the data source

    IMPORTANT: this may not be valid for the data source and will likely render
    them unusable in other tools, but should successfully roundtrip unless we
    disable writing using other encodings.

    NOTE: FlatGeobuff driver cannot handle non-UTF data in GDAL >= 3.9
    """
    encoding, text = encoded_text

    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")], dtype=object
    )

    field_data = [np.array([text], dtype=object)]

    fields = [text]
    meta = {"geometry_type": "Point", "crs": "EPSG:4326", "encoding": encoding}

    filename = tmp_path / f"test.{ext}"
    write(filename, geometry, field_data, fields, **meta)

    # cannot open these files without specifying encoding
    with pytest.raises(UnicodeDecodeError):
        read(filename)

    with pytest.raises(UnicodeDecodeError):
        read_info(filename)

    # must provide encoding to read these properly
    actual_meta, _, _, actual_field_data = read(filename, encoding=encoding)
    assert actual_meta["fields"][0] == text
    assert actual_field_data[0] == text
    assert read_info(filename, encoding=encoding)["fields"][0] == text


def test_non_utf8_encoding_io_shapefile(tmp_path, encoded_text):
    encoding, text = encoded_text

    # Point(0, 0)
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")], dtype=object
    )

    field_data = [np.array([text], dtype=object)]

    fields = [text]
    meta = {"geometry_type": "Point", "crs": "EPSG:4326", "encoding": encoding}

    filename = tmp_path / "test.shp"
    write(filename, geometry, field_data, fields, **meta)

    # NOTE: GDAL automatically creates a cpg file with the encoding name, which
    # means that if we read this without specifying the encoding it uses the
    # correct one
    actual_meta, _, _, actual_field_data = read(filename)
    assert actual_meta["fields"][0] == text
    assert actual_field_data[0] == text
    assert read_info(filename)["fields"][0] == text

    # verify that if cpg file is not present, that user-provided encoding must be used
    filename.with_suffix(".cpg").unlink()

    # We will assume ISO-8859-1, which is wrong
    miscoded = text.encode(encoding).decode("ISO-8859-1")
    bad_meta, _, _, bad_field_data = read(filename)
    assert bad_meta["fields"][0] == miscoded
    assert bad_field_data[0] == miscoded
    assert read_info(filename)["fields"][0] == miscoded

    # If encoding is provided, that should yield correct text
    actual_meta, _, _, actual_field_data = read(filename, encoding=encoding)
    assert actual_meta["fields"][0] == text
    assert actual_field_data[0] == text
    assert read_info(filename, encoding=encoding)["fields"][0] == text

    # verify that setting encoding does not corrupt SHAPE_ENCODING option if set
    # globally (it is ignored during read when encoding is specified by user)
    try:
        set_gdal_config_options({"SHAPE_ENCODING": "CP1254"})
        _ = read(filename, encoding=encoding)
        assert get_gdal_config_option("SHAPE_ENCODING") == "CP1254"

    finally:
        # reset to clear between tests
        set_gdal_config_options({"SHAPE_ENCODING": None})


def test_write_with_mask(tmp_path):
    # Point(0, 0), null
    geometry = np.array(
        [bytes.fromhex("010100000000000000000000000000000000000000")] * 3,
        dtype=object,
    )
    field_data = [np.array([1, 2, 3], dtype="int32")]
    field_mask = [np.array([False, True, False])]
    fields = ["col"]
    meta = {"geometry_type": "Point", "crs": "EPSG:4326"}

    filename = tmp_path / "test.geojson"
    write(filename, geometry, field_data, fields, field_mask, **meta)
    result_geometry, result_fields = read(filename)[2:]
    assert np.array_equal(result_geometry, geometry)
    np.testing.assert_allclose(result_fields[0], np.array([1, np.nan, 3]))

    # wrong length for mask
    field_mask = [np.array([False, True])]
    with pytest.raises(ValueError):
        write(filename, geometry, field_data, fields, field_mask, **meta)

    # wrong number of mask arrays
    field_mask = [np.array([False, True, False])] * 2
    with pytest.raises(ValueError):
        write(filename, geometry, field_data, fields, field_mask, **meta)


@requires_arrow_api
def test_open_arrow_capsule_protocol_without_pyarrow(naturalearth_lowres):
    # this test is included here instead of test_arrow.py to ensure we also run
    # it when pyarrow is not installed

    with open_arrow(naturalearth_lowres) as (meta, reader):
        assert isinstance(meta, dict)
        assert isinstance(reader, pyogrio._io._ArrowStream)
        capsule = reader.__arrow_c_stream__()
        assert (
            ctypes.pythonapi.PyCapsule_IsValid(
                ctypes.py_object(capsule), b"arrow_array_stream"
            )
            == 1
        )


@pytest.mark.skipif(HAS_PYARROW, reason="pyarrow is installed")
@requires_arrow_api
def test_open_arrow_error_no_pyarrow(naturalearth_lowres):
    # this test is included here instead of test_arrow.py to ensure we run
    # it when pyarrow is not installed

    with pytest.raises(ImportError):
        with open_arrow(naturalearth_lowres, use_pyarrow=True) as _:
            pass
