from pathlib import Path

import numpy as np
from numpy import allclose, array_equal

from pyogrio import (
    __gdal_geos_version__,
    __gdal_version__,
    detect_write_driver,
    get_gdal_config_option,
    get_gdal_data_path,
    list_drivers,
    list_layers,
    read_bounds,
    read_info,
    set_gdal_config_options,
    vsi_listtree,
    vsi_rmtree,
    vsi_unlink,
)
from pyogrio._compat import GDAL_GE_38
from pyogrio._env import GDALEnv
from pyogrio.errors import DataLayerError, DataSourceError
from pyogrio.raw import read, write
from pyogrio.tests.conftest import START_FID, prepare_testfile, requires_shapely

import pytest

with GDALEnv():
    # NOTE: this must be AFTER above imports, which init the GDAL and PROJ data
    # search paths
    from pyogrio._ogr import has_gdal_data, has_proj_data, ogr_driver_supports_write


try:
    import shapely
except ImportError:
    pass


def test_gdal_data():
    # test will fail if GDAL data files cannot be found, indicating an
    # installation error
    assert has_gdal_data()


def test_proj_data():
    # test will fail if PROJ data files cannot be found, indicating an
    # installation error
    assert has_proj_data()


def test_get_gdal_data_path():
    # test will fail if the function returns None, which means that GDAL
    # cannot find data files, indicating an installation error
    assert isinstance(get_gdal_data_path(), str)


def test_gdal_geos_version():
    assert __gdal_geos_version__ is None or isinstance(__gdal_geos_version__, tuple)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("test.shp", "ESRI Shapefile"),
        ("test.shp.zip", "ESRI Shapefile"),
        ("test.geojson", "GeoJSON"),
        ("test.geojsonl", "GeoJSONSeq"),
        ("test.gpkg", "GPKG"),
        pytest.param(
            "test.gpkg.zip",
            "GPKG",
            marks=pytest.mark.skipif(
                __gdal_version__ < (3, 7, 0),
                reason="writing *.gpkg.zip requires GDAL >= 3.7.0",
            ),
        ),
        # postgres can be detected by prefix instead of extension
        pytest.param(
            "PG:dbname=test",
            "PostgreSQL",
            marks=pytest.mark.skipif(
                "PostgreSQL" not in list_drivers(),
                reason="PostgreSQL path test requires PostgreSQL driver",
            ),
        ),
    ],
)
def test_detect_write_driver(path, expected):
    assert detect_write_driver(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "test.svg",  # only supports read
        "test.",  # not a valid extension
        "test",  # no extension or prefix
        "test.foo",  # not a valid extension
        "FOO:test",  # not a valid prefix
    ],
)
def test_detect_write_driver_unsupported(path):
    with pytest.raises(ValueError, match="Could not infer driver from path"):
        detect_write_driver(path)


@pytest.mark.parametrize("path", ["test.xml", "test.txt"])
def test_detect_write_driver_multiple_unsupported(path):
    with pytest.raises(ValueError, match="multiple drivers are available"):
        detect_write_driver(path)


@pytest.mark.parametrize(
    "driver,expected",
    [
        # drivers known to be well-supported by pyogrio
        ("ESRI Shapefile", True),
        ("GeoJSON", True),
        ("GeoJSONSeq", True),
        ("GPKG", True),
        # drivers not supported for write by GDAL
        ("HTTP", False),
        ("OAPIF", False),
    ],
)
def test_ogr_driver_supports_write(driver, expected):
    assert ogr_driver_supports_write(driver) == expected


def test_list_drivers():
    all_drivers = list_drivers()

    # verify that the core drivers are present
    for name in ("ESRI Shapefile", "GeoJSON", "GeoJSONSeq", "GPKG", "OpenFileGDB"):
        assert name in all_drivers

        expected_capability = "rw"
        if name == "OpenFileGDB" and __gdal_version__ < (3, 6, 0):
            expected_capability = "r"

        assert all_drivers[name] == expected_capability

    drivers = list_drivers(read=True)
    expected = {k: v for k, v in all_drivers.items() if v.startswith("r")}
    assert len(drivers) == len(expected)

    drivers = list_drivers(write=True)
    expected = {k: v for k, v in all_drivers.items() if v.endswith("w")}
    assert len(drivers) == len(expected)

    drivers = list_drivers(read=True, write=True)
    expected = {
        k: v for k, v in all_drivers.items() if v.startswith("r") and v.endswith("w")
    }
    assert len(drivers) == len(expected)


def test_list_layers(
    naturalearth_lowres,
    naturalearth_lowres_vsi,
    naturalearth_lowres_vsimem,
    line_zm_file,
    curve_file,
    curve_polygon_file,
    multisurface_file,
    no_geometry_file,
):
    assert array_equal(
        list_layers(naturalearth_lowres), [["naturalearth_lowres", "Polygon"]]
    )

    assert array_equal(
        list_layers(naturalearth_lowres_vsi[1]), [["naturalearth_lowres", "Polygon"]]
    )

    assert array_equal(
        list_layers(naturalearth_lowres_vsimem),
        [["naturalearth_lowres", "MultiPolygon"]],
    )

    # Measured 3D is downgraded to plain 3D during read
    # Make sure this warning is raised
    with pytest.warns(
        UserWarning, match=r"Measured \(M\) geometry types are not supported"
    ):
        assert array_equal(list_layers(line_zm_file), [["line_zm", "LineString Z"]])

    # Curve / surface types are downgraded to plain types
    assert array_equal(list_layers(curve_file), [["curve", "LineString"]])
    assert array_equal(list_layers(curve_polygon_file), [["curvepolygon", "Polygon"]])
    assert array_equal(
        list_layers(multisurface_file), [["multisurface", "MultiPolygon"]]
    )

    # Make sure that nonspatial layer has None for geometry
    assert array_equal(list_layers(no_geometry_file), [["no_geometry", None]])


def test_list_layers_bytes(geojson_bytes):
    layers = list_layers(geojson_bytes)

    assert layers.shape == (1, 2)
    assert layers[0, 0] == "test"


def test_list_layers_nonseekable_bytes(nonseekable_bytes):
    layers = list_layers(nonseekable_bytes)

    assert layers.shape == (1, 2)
    assert layers[0, 1] == "Point"


def test_list_layers_filelike(geojson_filelike):
    layers = list_layers(geojson_filelike)

    assert layers.shape == (1, 2)
    assert layers[0, 0] == "test"


@pytest.mark.parametrize(
    "testfile",
    ["naturalearth_lowres", "naturalearth_lowres_vsimem", "naturalearth_lowres_vsi"],
)
def test_read_bounds(testfile, request):
    path = request.getfixturevalue(testfile)
    path = path if not isinstance(path, tuple) else path[1]

    fids, bounds = read_bounds(path)
    assert fids.shape == (177,)
    assert bounds.shape == (4, 177)
    assert fids[0] == START_FID[Path(path).suffix]
    # Fiji; wraps antimeridian
    assert allclose(bounds[:, 0], [-180.0, -18.28799, 180.0, -16.02088])


def test_read_bounds_bytes(geojson_bytes):
    fids, bounds = read_bounds(geojson_bytes)
    assert fids.shape == (3,)
    assert bounds.shape == (4, 3)
    assert allclose(bounds[:, 0], [-180.0, -18.28799, 180.0, -16.02088])


def test_read_bounds_nonseekable_bytes(nonseekable_bytes):
    fids, bounds = read_bounds(nonseekable_bytes)
    assert fids.shape == (1,)
    assert bounds.shape == (4, 1)
    assert allclose(bounds[:, 0], [1, 1, 1, 1])


def test_read_bounds_filelike(geojson_filelike):
    fids, bounds = read_bounds(geojson_filelike)
    assert fids.shape == (3,)
    assert bounds.shape == (4, 3)
    assert allclose(bounds[:, 0], [-180.0, -18.28799, 180.0, -16.02088])


def test_read_bounds_max_features(naturalearth_lowres):
    bounds = read_bounds(naturalearth_lowres, max_features=2)[1]
    assert bounds.shape == (4, 2)


def test_read_bounds_unspecified_layer_warning(data_dir):
    """Reading a multi-layer file without specifying a layer gives a warning."""
    with pytest.warns(UserWarning, match="More than one layer found "):
        read_bounds(data_dir / "sample.osm.pbf")


def test_read_bounds_negative_max_features(naturalearth_lowres):
    with pytest.raises(ValueError, match="'max_features' must be >= 0"):
        read_bounds(naturalearth_lowres, max_features=-1)


def test_read_bounds_skip_features(naturalearth_lowres):
    expected_bounds = read_bounds(naturalearth_lowres, max_features=11)[1][:, 10]
    fids, bounds = read_bounds(naturalearth_lowres, skip_features=10)
    assert bounds.shape == (4, 167)
    assert allclose(bounds[:, 0], expected_bounds)
    assert fids[0] == 10


def test_read_bounds_negative_skip_features(naturalearth_lowres):
    with pytest.raises(ValueError, match="'skip_features' must be >= 0"):
        read_bounds(naturalearth_lowres, skip_features=-1)


def test_read_bounds_where_invalid(naturalearth_lowres_all_ext):
    with pytest.raises(ValueError, match="Invalid SQL"):
        read_bounds(naturalearth_lowres_all_ext, where="invalid")


def test_read_bounds_where(naturalearth_lowres):
    fids, bounds = read_bounds(naturalearth_lowres, where="iso_a3 = 'CAN'")
    assert fids.shape == (1,)
    assert bounds.shape == (4, 1)
    assert fids[0] == 3
    assert allclose(bounds[:, 0], [-140.99778, 41.675105, -52.648099, 83.23324])


@pytest.mark.parametrize("bbox", [(1,), (1, 2), (1, 2, 3)])
def test_read_bounds_bbox_invalid(naturalearth_lowres, bbox):
    with pytest.raises(ValueError, match="Invalid bbox"):
        read_bounds(naturalearth_lowres, bbox=bbox)


def test_read_bounds_bbox(naturalearth_lowres_all_ext):
    # should return no features
    fids, bounds = read_bounds(
        naturalearth_lowres_all_ext, bbox=(0, 0, 0.00001, 0.00001)
    )

    assert fids.shape == (0,)
    assert bounds.shape == (4, 0)

    fids, bounds = read_bounds(naturalearth_lowres_all_ext, bbox=(-85, 8, -80, 10))

    assert fids.shape == (2,)
    fids_expected = np.array([33, 34])  # PAN, CRI
    fids_expected += START_FID[naturalearth_lowres_all_ext.suffix]
    assert array_equal(fids, fids_expected)

    assert bounds.shape == (4, 2)
    assert allclose(
        bounds.T,
        [
            [-82.96578305, 7.22054149, -77.24256649, 9.61161001],
            [-85.94172543, 8.22502798, -82.54619626, 11.21711925],
        ],
    )


@requires_shapely
@pytest.mark.parametrize(
    "mask",
    [
        {"type": "Point", "coordinates": [0, 0]},
        '{"type": "Point", "coordinates": [0, 0]}',
        "invalid",
    ],
)
def test_read_bounds_mask_invalid(naturalearth_lowres, mask):
    with pytest.raises(ValueError, match="'mask' parameter must be a Shapely geometry"):
        read_bounds(naturalearth_lowres, mask=mask)


@requires_shapely
def test_read_bounds_bbox_mask_invalid(naturalearth_lowres):
    with pytest.raises(ValueError, match="cannot set both 'bbox' and 'mask'"):
        read_bounds(
            naturalearth_lowres, bbox=(-85, 8, -80, 10), mask=shapely.Point(-105, 55)
        )


@requires_shapely
@pytest.mark.parametrize(
    "mask,expected",
    [
        ("POINT (-105 55)", [3]),
        ("POLYGON ((-80 8, -80 10, -85 10, -85 8, -80 8))", [33, 34]),
        (
            """POLYGON ((
                6.101929 50.97085,
                5.773002 50.906611,
                5.593156 50.642649,
                6.059271 50.686052,
                6.374064 50.851481,
                6.101929 50.97085
            ))""",
            [121, 129, 130],
        ),
        (
            """GEOMETRYCOLLECTION (
                POINT (-7.7 53),
                POLYGON ((-80 8, -80 10, -85 10, -85 8, -80 8))
            )""",
            [33, 34, 133],
        ),
    ],
)
def test_read_bounds_mask(naturalearth_lowres_all_ext, mask, expected):
    mask = shapely.from_wkt(mask)

    fids = read_bounds(naturalearth_lowres_all_ext, mask=mask)[0]

    fids_expected = np.array(expected) + START_FID[naturalearth_lowres_all_ext.suffix]
    assert array_equal(fids, fids_expected)


@pytest.mark.skipif(
    __gdal_version__ < (3, 4, 0),
    reason="Cannot determine if GEOS is present or absent for GDAL < 3.4",
)
def test_read_bounds_bbox_intersects_vs_envelope_overlaps(naturalearth_lowres_all_ext):
    # If GEOS is present and used by GDAL, bbox filter will be based on intersection
    # of bbox and actual geometries; if GEOS is absent or not used by GDAL, it
    # will be based on overlap of bounding boxes instead
    fids, _ = read_bounds(naturalearth_lowres_all_ext, bbox=(-140, 20, -100, 45))

    if __gdal_geos_version__ is None:
        # bboxes for CAN, RUS overlap but do not intersect geometries
        assert fids.shape == (4,)
        fids_expected = np.array([3, 4, 18, 27])  # CAN, USA, RUS, MEX
        fids_expected += START_FID[naturalearth_lowres_all_ext.suffix]
        assert array_equal(fids, fids_expected)

    else:
        assert fids.shape == (2,)
        fids_expected = np.array([4, 27])  # USA, MEX
        fids_expected += START_FID[naturalearth_lowres_all_ext.suffix]
        assert array_equal(fids, fids_expected)


@pytest.mark.parametrize("naturalearth_lowres", [".shp", ".gpkg"], indirect=True)
def test_read_info(naturalearth_lowres):
    meta = read_info(naturalearth_lowres)

    assert meta["layer_name"] == "naturalearth_lowres"
    assert meta["crs"] == "EPSG:4326"
    assert meta["encoding"] == "UTF-8"
    assert meta["fields"].shape == (5,)
    assert meta["dtypes"].tolist() == ["int64", "object", "object", "object", "float64"]
    assert meta["features"] == 177
    assert allclose(meta["total_bounds"], (-180, -90, 180, 83.64513))
    assert meta["capabilities"]["random_read"] is True
    assert meta["capabilities"]["fast_spatial_filter"] is False
    assert meta["capabilities"]["fast_feature_count"] is True
    assert meta["capabilities"]["fast_total_bounds"] is True

    if naturalearth_lowres.suffix == ".gpkg":
        assert meta["fid_column"] == "fid"
        assert meta["geometry_name"] == "geom"
        assert meta["geometry_type"] == "MultiPolygon"
        assert meta["driver"] == "GPKG"
        if GDAL_GE_38:
            # this capability is only True for GPKG if GDAL >= 3.8
            assert meta["capabilities"]["fast_set_next_by_index"] is True
    elif naturalearth_lowres.suffix == ".shp":
        # fid_column == "" for formats where fid is not physically stored
        assert meta["fid_column"] == ""
        # geometry_name == "" for formats where geometry column name cannot be
        # customized
        assert meta["geometry_name"] == ""
        assert meta["geometry_type"] == "Polygon"
        assert meta["driver"] == "ESRI Shapefile"
        assert meta["capabilities"]["fast_set_next_by_index"] is True
    else:
        raise ValueError(f"test not implemented for ext {naturalearth_lowres.suffix}")


@pytest.mark.parametrize(
    "testfile", ["naturalearth_lowres_vsimem", "naturalearth_lowres_vsi"]
)
def test_read_info_vsi(testfile, request):
    path = request.getfixturevalue(testfile)
    path = path if not isinstance(path, tuple) else path[1]

    meta = read_info(path)

    assert meta["fields"].shape == (5,)
    assert meta["features"] == 177


def test_read_info_bytes(geojson_bytes):
    meta = read_info(geojson_bytes)

    assert meta["fields"].shape == (5,)
    assert meta["features"] == 3


def test_read_info_nonseekable_bytes(nonseekable_bytes):
    meta = read_info(nonseekable_bytes)

    assert meta["fields"].shape == (0,)
    assert meta["features"] == 1


def test_read_info_filelike(geojson_filelike):
    meta = read_info(geojson_filelike)

    assert meta["fields"].shape == (5,)
    assert meta["features"] == 3


@pytest.mark.parametrize(
    "dataset_kwargs,fields",
    [
        ({}, ["top_level", "intermediate_level"]),
        (
            {"FLATTEN_NESTED_ATTRIBUTES": "YES"},
            [
                "top_level",
                "intermediate_level_bottom_level",
            ],
        ),
        (
            {"flatten_nested_attributes": "yes"},
            [
                "top_level",
                "intermediate_level_bottom_level",
            ],
        ),
        (
            {"flatten_nested_attributes": True},
            [
                "top_level",
                "intermediate_level_bottom_level",
            ],
        ),
    ],
)
def test_read_info_dataset_kwargs(nested_geojson_file, dataset_kwargs, fields):
    meta = read_info(nested_geojson_file, **dataset_kwargs)
    assert meta["fields"].tolist() == fields


def test_read_info_invalid_dataset_kwargs(naturalearth_lowres):
    with pytest.warns(RuntimeWarning, match="does not support open option INVALID"):
        read_info(naturalearth_lowres, INVALID="YES")


def test_read_info_force_feature_count_exception(data_dir):
    with pytest.raises(DataLayerError, match="Could not iterate over features"):
        read_info(data_dir / "sample.osm.pbf", layer="lines", force_feature_count=True)


@pytest.mark.parametrize(
    "layer, force, expected",
    [
        ("points", False, -1),
        ("points", True, 8),
        ("lines", False, -1),
        ("lines", True, 36),
    ],
)
def test_read_info_force_feature_count(data_dir, layer, force, expected):
    # the sample OSM file has non-increasing node IDs which causes the default
    # custom indexing to raise an exception iterating over features
    meta = read_info(
        data_dir / "sample.osm.pbf",
        layer=layer,
        force_feature_count=force,
        USE_CUSTOM_INDEXING=False,
    )
    assert meta["features"] == expected


@pytest.mark.parametrize(
    "force_total_bounds, expected_total_bounds",
    [(True, (-180.0, -90.0, 180.0, 83.64513)), (False, None)],
)
def test_read_info_force_total_bounds(
    tmp_path, naturalearth_lowres, force_total_bounds, expected_total_bounds
):
    geojson_path = prepare_testfile(
        naturalearth_lowres, dst_dir=tmp_path, ext=".geojsonl"
    )

    info = read_info(geojson_path, force_total_bounds=force_total_bounds)
    if expected_total_bounds is not None:
        assert allclose(info["total_bounds"], expected_total_bounds)
    else:
        assert info["total_bounds"] is None


def test_read_info_unspecified_layer_warning(data_dir):
    """Reading a multi-layer file without specifying a layer gives a warning."""
    with pytest.warns(UserWarning, match="More than one layer found "):
        read_info(data_dir / "sample.osm.pbf")


def test_read_info_without_geometry(no_geometry_file):
    assert read_info(no_geometry_file)["total_bounds"] is None


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("CPL_DEBUG", "ON", True),
        ("CPL_DEBUG", True, True),
        ("CPL_DEBUG", "OFF", False),
        ("CPL_DEBUG", False, False),
    ],
)
def test_set_config_options(name, value, expected):
    set_gdal_config_options({name: value})
    actual = get_gdal_config_option(name)
    assert actual == expected


def test_reset_config_options():
    set_gdal_config_options({"foo": "bar"})
    assert get_gdal_config_option("foo") == "bar"

    set_gdal_config_options({"foo": None})
    assert get_gdal_config_option("foo") is None


def test_error_handling(capfd):
    # an operation that triggers a GDAL Failure
    # -> error translated into Python exception + not printed to stderr
    with pytest.raises(DataSourceError, match="No such file or directory"):
        read_info("non-existent.shp")

    assert capfd.readouterr().err == ""


def test_error_handling_warning(capfd, naturalearth_lowres):
    # an operation that triggers a GDAL Warning
    # -> translated into a Python warning + not printed to stderr
    with pytest.warns(RuntimeWarning, match="does not support open option INVALID"):
        read_info(naturalearth_lowres, INVALID="YES")

    assert capfd.readouterr().err == ""


def test_vsimem_listtree_rmtree_unlink(naturalearth_lowres):
    """Test all basic functionalities of file handling in /vsimem/."""
    # Prepare test data in /vsimem
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta["spatial_index"] = False
    meta["geometry_type"] = "MultiPolygon"
    test_file_path = Path("/vsimem/pyogrio_test_naturalearth_lowres.gpkg")
    test_dir_path = Path(f"/vsimem/pyogrio_dir_test/{naturalearth_lowres.stem}.gpkg")

    write(test_file_path, geometry, field_data, **meta)
    write(test_dir_path, geometry, field_data, **meta)

    # Check if everything was created properly with listtree
    files = vsi_listtree("/vsimem/")
    assert test_file_path.as_posix() in files
    assert test_dir_path.as_posix() in files

    # Check listtree with pattern
    files = vsi_listtree("/vsimem/", pattern="pyogrio_dir_test*.gpkg")
    assert test_file_path.as_posix() not in files
    assert test_dir_path.as_posix() in files

    files = vsi_listtree("/vsimem/", pattern="pyogrio_test*.gpkg")
    assert test_file_path.as_posix() in files
    assert test_dir_path.as_posix() not in files

    # Remove test_dir and its contents
    vsi_rmtree(test_dir_path.parent)
    files = vsi_listtree("/vsimem/")
    assert test_file_path.as_posix() in files
    assert test_dir_path.as_posix() not in files

    # Remove test_file
    vsi_unlink(test_file_path)


def test_vsimem_rmtree_error(naturalearth_lowres_vsimem):
    with pytest.raises(NotADirectoryError, match="Path is not a directory"):
        vsi_rmtree(naturalearth_lowres_vsimem)

    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        vsi_rmtree("/vsimem/non-existent")

    with pytest.raises(
        OSError, match="path to in-memory file or directory is required"
    ):
        vsi_rmtree("/vsimem")
    with pytest.raises(
        OSError, match="path to in-memory file or directory is required"
    ):
        vsi_rmtree("/vsimem/")

    # Verify that naturalearth_lowres_vsimem still exists.
    assert naturalearth_lowres_vsimem.as_posix() in vsi_listtree("/vsimem")


def test_vsimem_unlink_error(naturalearth_lowres_vsimem):
    with pytest.raises(IsADirectoryError, match="Path is a directory"):
        vsi_unlink(naturalearth_lowres_vsimem.parent)

    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        vsi_unlink("/vsimem/non-existent.gpkg")
