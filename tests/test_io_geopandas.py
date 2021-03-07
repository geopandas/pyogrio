import os
import pandas as pd
import geopandas as gp
from pandas.testing import assert_frame_equal
from geopandas.testing import assert_geodataframe_equal
import pytest

from pyogrio import list_layers
from pyogrio.geopandas import read_dataframe, write_dataframe


def test_read_dataframe(naturalearth_lowres):
    df = read_dataframe(naturalearth_lowres)

    assert isinstance(df, gp.GeoDataFrame)

    assert df.crs == "EPSG:4326"
    assert len(df) == 177
    assert df.columns.tolist() == [
        "pop_est",
        "continent",
        "name",
        "iso_a3",
        "gdp_md_est",
        "geometry",
    ]

    assert df.geometry.iloc[0].type == "MultiPolygon"


def test_read_dataframe_vsi(naturalearth_lowres_vsi):
    df = read_dataframe(naturalearth_lowres_vsi)
    assert len(df) == 177


def test_read_no_geometry(naturalearth_lowres):
    df = read_dataframe(naturalearth_lowres, read_geometry=False)
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df, gp.GeoDataFrame)


# def test_read_force_2d(nhd_hr):
#     df = read_dataframe(nhd_hr, layer="NHDFlowline", max_features=1)
#     assert df.iloc[0].geometry.has_z

#     df = read_dataframe(nhd_hr, layer="NHDFlowline", force_2d=True, max_features=1)
#     assert not df.iloc[0].geometry.has_z


# def test_read_layer(nhd_hr):
#     layers = list_layers(nhd_hr)
#     # The first layer is read by default (NOTE: first layer has no features)
#     df = read_dataframe(nhd_hr, read_geometry=False, max_features=1)
#     df2 = read_dataframe(
#         nhd_hr, layer=layers[0][0], read_geometry=False, max_features=1
#     )
#     assert_frame_equal(df, df2)

#     # Reading a specific layer should return that layer.
#     # Detected here by a known column.
#     df = read_dataframe(nhd_hr, layer="WBDHU2", read_geometry=False, max_features=1)
#     assert "HUC2" in df.columns


# def test_read_datetime(nhd_hr):
#     df = read_dataframe(nhd_hr, max_features=1)
#     assert df.ExternalIDEntryDate.dtype.name == "datetime64[ns]"


# def test_read_null_values(naturalearth_lowres):
#     df = read_dataframe(naturalearth_lowres, read_geometry=False)

#     # make sure that Null values are preserved
#     assert df.NAME_ZH.isnull().max() == True
#     assert df.loc[df.NAME_ZH.isnull()].NAME_ZH.iloc[0] == None


def test_read_where(naturalearth_lowres):
    # empty filter should return full set of records
    df = read_dataframe(naturalearth_lowres, where="")
    assert len(df) == 177

    # should return singular item
    df = read_dataframe(naturalearth_lowres, where="iso_a3 = 'CAN'")
    assert len(df) == 1
    assert df.iloc[0].iso_a3 == "CAN"

    df = read_dataframe(naturalearth_lowres, where="iso_a3 IN ('CAN', 'USA', 'MEX')")
    assert len(df) == 3
    assert len(set(df.iso_a3.unique()).difference(["CAN", "USA", "MEX"])) == 0

    # should return items within range
    df = read_dataframe(
        naturalearth_lowres, where="POP_EST >= 10000000 AND POP_EST < 100000000"
    )
    assert len(df) == 75
    assert df.pop_est.min() >= 10000000
    assert df.pop_est.max() < 100000000

    # should match no items
    df = read_dataframe(naturalearth_lowres, where="ISO_A3 = 'INVALID'")
    assert len(df) == 0


def test_read_where_invalid(naturalearth_lowres):
    with pytest.raises(ValueError, match="Invalid SQL"):
        read_dataframe(naturalearth_lowres, where="invalid")


# def test_write_dataframe(tmpdir, naturalearth_lowres):
#     expected = read_dataframe(naturalearth_lowres)

#     filename = os.path.join(str(tmpdir), "test.shp")
#     write_dataframe(expected, filename)

#     assert os.path.exists(filename)

#     df = read_dataframe(filename)
#     assert_geodataframe_equal(df, expected)


@pytest.mark.parametrize(
    "driver,ext",
    [
        ("ESRI Shapefile", "shp"),
        ("GeoJSON", "geojson"),
        ("GeoJSONSeq", "geojsons"),
        ("GPKG", "gpkg"),
    ],
)
def test_write_dataframe(tmpdir, naturalearth_lowres, driver, ext):
    expected = read_dataframe(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), f"test.{ext}")
    write_dataframe(expected, filename, driver=driver)

    assert os.path.exists(filename)

    df = read_dataframe(filename)

    if driver != "GeoJSONSeq":
        # GeoJSONSeq driver I/O reorders features and / or vertices, and does
        # not support roundtrip comparison

        # Coordinates are not precisely equal when written to JSON
        # dtypes do not necessarily round-trip precisely through JSON
        is_json = driver == "GeoJSON"

        assert_geodataframe_equal(
            df, expected, check_less_precise=is_json, check_dtype=not is_json
        )


# def test_write_dataframe_nhd(tmpdir, nhd_hr):
#     df = read_dataframe(nhd_hr, layer="NHDFlowline", max_features=2)

#     # Datetime not currently supported
#     df = df.drop(columns="FDate")

#     filename = os.path.join(str(tmpdir), "test.shp")
#     write_dataframe(df, filename)

