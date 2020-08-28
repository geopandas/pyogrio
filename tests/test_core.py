from numpy import array_equal
import pytest

from pyogrio import list_layers, read_info


def test_list_layers(naturalearth_lowres, naturalearth_modres, nhd_wbd, nhd_hr):
    assert array_equal(
        list_layers(naturalearth_lowres), [["ne_110m_admin_0_countries", "Polygon"]]
    )

    assert array_equal(
        list_layers(naturalearth_modres), [["ne_10m_admin_0_countries", "Polygon"]]
    )

    wbd_layers = list_layers(nhd_wbd)
    assert len(wbd_layers) == 20
    assert array_equal(wbd_layers[7], ["WBDLine", "MultiLineString"])
    assert array_equal(wbd_layers[8], ["WBDHU8", "MultiPolygon"])

    # Measured 3D is downgraded to 2.5D during read
    # Make sure this warning is raised
    with pytest.warns(
        UserWarning, match=r"Measured \(M\) geometry types are not supported"
    ):
        hr_layers = list_layers(nhd_hr)
        assert len(hr_layers) == 75

        # Make sure that nonspatial layer has None for geometry
        assert array_equal(hr_layers[0], ["ExternalCrosswalk", None])

        # Confirm that measured 3D is downgraded to 2.5D during read
        assert array_equal(hr_layers[54], ["NHDArea", "2.5D MultiPolygon"])
        assert array_equal(hr_layers[55], ["NHDFlowline", "2.5D MultiLineString"])


def test_read_info(naturalearth_lowres):
    meta = read_info(naturalearth_lowres)

    assert meta["crs"] == "EPSG:4326"
    assert meta["geometry_type"] == "Polygon"
    assert meta["encoding"] == "UTF-8"
    assert meta["fields"].shape == (94,)
    assert meta["features"] == 177
