from numpy import array_equal, allclose
import pytest

from pyogrio import (
    __gdal_version__,
    __gdal_geos_version__,
    list_drivers,
    list_layers,
    read_bounds,
    read_info,
    set_gdal_config_options,
    get_gdal_config_option,
    get_gdal_data_path,
)

from pyogrio._env import GDALEnv

with GDALEnv():
    # NOTE: this must be AFTER above imports, which init the GDAL and PROJ data
    # search paths
    from pyogrio._ogr import ogr_driver_supports_write, has_gdal_data, has_proj_data


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
        # drivers currently unsupported for write even though GDAL can write them
        ("XLSX", False),
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


def test_list_layers(naturalearth_lowres, naturalearth_lowres_vsi, test_fgdb_vsi):
    assert array_equal(
        list_layers(naturalearth_lowres), [["naturalearth_lowres", "Polygon"]]
    )

    assert array_equal(
        list_layers(naturalearth_lowres_vsi[1]), [["naturalearth_lowres", "Polygon"]]
    )

    # Measured 3D is downgraded to 2.5D during read
    # Make sure this warning is raised
    with pytest.warns(
        UserWarning, match=r"Measured \(M\) geometry types are not supported"
    ):
        fgdb_layers = list_layers(test_fgdb_vsi)
        # GDAL >= 3.4.0 includes 'another_relationship' layer
        assert len(fgdb_layers) >= 7

        # Make sure that nonspatial layer has None for geometry
        assert array_equal(fgdb_layers[0], ["basetable_2", None])

        # Confirm that measured 3D is downgraded to 2.5D during read
        assert array_equal(fgdb_layers[3], ["test_lines", "2.5D MultiLineString"])
        assert array_equal(fgdb_layers[6], ["test_areas", "2.5D MultiPolygon"])


def test_read_bounds(naturalearth_lowres):
    fids, bounds = read_bounds(naturalearth_lowres)
    assert fids.shape == (177,)
    assert bounds.shape == (4, 177)

    assert fids[0] == 0
    # Fiji; wraps antimeridian
    assert allclose(bounds[:, 0], [-180.0, -18.28799, 180.0, -16.02088])


def test_read_bounds_max_features(naturalearth_lowres):
    bounds = read_bounds(naturalearth_lowres, max_features=2)[1]
    assert bounds.shape == (4, 2)


def test_read_bounds_skip_features(naturalearth_lowres):
    expected_bounds = read_bounds(naturalearth_lowres, max_features=11)[1][:, 10]
    fids, bounds = read_bounds(naturalearth_lowres, skip_features=10)
    assert bounds.shape == (4, 167)
    assert allclose(bounds[:, 0], expected_bounds)
    assert fids[0] == 10


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
    with pytest.warns(UserWarning, match="does not have any features to read"):
        fids, bounds = read_bounds(
            naturalearth_lowres_all_ext, bbox=(0, 0, 0.00001, 0.00001)
        )

    assert fids.shape == (0,)
    assert bounds.shape == (4, 0)

    fids, bounds = read_bounds(naturalearth_lowres_all_ext, bbox=(-85, 8, -80, 10))

    assert fids.shape == (2,)
    if naturalearth_lowres_all_ext.suffix == ".gpkg":
        # fid in gpkg is 1-based
        assert array_equal(fids, [34, 35])  # PAN, CRI
    else:
        # fid in other formats is 0-based
        assert array_equal(fids, [33, 34])  # PAN, CRI

    assert bounds.shape == (4, 2)
    assert allclose(
        bounds.T,
        [
            [-82.96578305, 7.22054149, -77.24256649, 9.61161001],
            [-85.94172543, 8.22502798, -82.54619626, 11.21711925],
        ],
    )


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
        if naturalearth_lowres_all_ext.suffix == ".gpkg":
            # fid in gpkg is 1-based
            assert array_equal(fids, [4, 5, 19, 28])  # CAN, USA, RUS, MEX
        else:
            # fid in other formats is 0-based
            assert array_equal(fids, [3, 4, 18, 27])  # CAN, USA, RUS, MEX

    else:
        assert fids.shape == (2,)
        if naturalearth_lowres_all_ext.suffix == ".gpkg":
            # fid in gpkg is 1-based
            assert array_equal(fids, [5, 28])  # USA, MEX
        else:
            # fid in other formats is 0-based
            assert array_equal(fids, [4, 27])  # USA, MEX


def test_read_info(naturalearth_lowres):
    meta = read_info(naturalearth_lowres)

    assert meta["crs"] == "EPSG:4326"
    assert meta["geometry_type"] == "Polygon"
    assert meta["encoding"] == "UTF-8"
    assert meta["fields"].shape == (5,)
    assert meta["dtypes"].tolist() == ["int64", "object", "object", "object", "float64"]
    assert meta["features"] == 177


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
