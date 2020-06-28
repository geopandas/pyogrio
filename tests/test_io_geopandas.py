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
        "featurecla",
        "scalerank",
        "LABELRANK",
        "SOVEREIGNT",
        "SOV_A3",
        "ADM0_DIF",
        "LEVEL",
        "TYPE",
        "ADMIN",
        "ADM0_A3",
        "GEOU_DIF",
        "GEOUNIT",
        "GU_A3",
        "SU_DIF",
        "SUBUNIT",
        "SU_A3",
        "BRK_DIFF",
        "NAME",
        "NAME_LONG",
        "BRK_A3",
        "BRK_NAME",
        "BRK_GROUP",
        "ABBREV",
        "POSTAL",
        "FORMAL_EN",
        "FORMAL_FR",
        "NAME_CIAWF",
        "NOTE_ADM0",
        "NOTE_BRK",
        "NAME_SORT",
        "NAME_ALT",
        "MAPCOLOR7",
        "MAPCOLOR8",
        "MAPCOLOR9",
        "MAPCOLOR13",
        "POP_EST",
        "POP_RANK",
        "GDP_MD_EST",
        "POP_YEAR",
        "LASTCENSUS",
        "GDP_YEAR",
        "ECONOMY",
        "INCOME_GRP",
        "WIKIPEDIA",
        "FIPS_10_",
        "ISO_A2",
        "ISO_A3",
        "ISO_A3_EH",
        "ISO_N3",
        "UN_A3",
        "WB_A2",
        "WB_A3",
        "WOE_ID",
        "WOE_ID_EH",
        "WOE_NOTE",
        "ADM0_A3_IS",
        "ADM0_A3_US",
        "ADM0_A3_UN",
        "ADM0_A3_WB",
        "CONTINENT",
        "REGION_UN",
        "SUBREGION",
        "REGION_WB",
        "NAME_LEN",
        "LONG_LEN",
        "ABBREV_LEN",
        "TINY",
        "HOMEPART",
        "MIN_ZOOM",
        "MIN_LABEL",
        "MAX_LABEL",
        "NE_ID",
        "WIKIDATAID",
        "NAME_AR",
        "NAME_BN",
        "NAME_DE",
        "NAME_EN",
        "NAME_ES",
        "NAME_FR",
        "NAME_EL",
        "NAME_HI",
        "NAME_HU",
        "NAME_ID",
        "NAME_IT",
        "NAME_JA",
        "NAME_KO",
        "NAME_NL",
        "NAME_PL",
        "NAME_PT",
        "NAME_RU",
        "NAME_SV",
        "NAME_TR",
        "NAME_VI",
        "NAME_ZH",
        "geometry",
    ]

    assert df.geometry.iloc[0].type == "MultiPolygon"


def test_read_dataframe_vsi(naturalearth_modres_vsi):
    df = read_dataframe(naturalearth_modres_vsi)
    assert len(df) == 255


def test_read_no_geometry(naturalearth_modres):
    df = read_dataframe(naturalearth_modres, read_geometry=False)
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df, gp.GeoDataFrame)


def test_read_layer(nhd_hr):
    layers = list_layers(nhd_hr)
    # The first layer is read by default (NOTE: first layer has no features)
    df = read_dataframe(nhd_hr, read_geometry=False, max_features=1)
    df2 = read_dataframe(
        nhd_hr, layer=layers[0][0], read_geometry=False, max_features=1
    )
    assert_frame_equal(df, df2)

    # Reading a specific layer should return that layer.
    # Detected here by a known column.
    df = read_dataframe(nhd_hr, layer="WBDHU2", read_geometry=False, max_features=1)
    assert "HUC2" in df.columns


def test_read_datetime(nhd_hr):
    df = read_dataframe(nhd_hr, max_features=1)
    assert df.ExternalIDEntryDate.dtype.name == "datetime64[ns]"


def test_read_null_values(naturalearth_modres):
    df = read_dataframe(naturalearth_modres, read_geometry=False)

    # make sure that Null values are preserved
    assert df.NAME_ZH.isnull().max() == True
    assert df.loc[df.NAME_ZH.isnull()].NAME_ZH.iloc[0] == None


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

    # Coordinates are not precisely equal when written to JSON
    # dtypes do not necessarily round-trip precisely through JSON
    is_json = driver in ("GeoJSON", "GeoJSONSeq")

    assert_geodataframe_equal(
        df, expected, check_less_precise=is_json, check_dtype=not is_json
    )


def test_write_dataframe_nhd(tmpdir, nhd_hr):
    df = read_dataframe(nhd_hr, layer="NHDFlowline", max_features=2)

    # Datetime not currently supported
    df = df.drop(columns="FDate")

    filename = os.path.join(str(tmpdir), "test.shp")
    write_dataframe(df, filename)

