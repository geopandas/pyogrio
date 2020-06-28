import json
import os
from pathlib import Path

import numpy as np
from numpy import array_equal

from pyogrio import read, read_info, list_layers, write


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

    hr_layers = list_layers(nhd_hr)
    assert len(hr_layers) == 75
    assert array_equal(hr_layers[54], ["NHDArea", "2.5D MultiPolygon"])
    # Measured 3D is downgraded to 2.5D during read
    assert array_equal(hr_layers[55], ["NHDFlowline", "2.5D MultiLineString"])


def test_read(naturalearth_lowres):
    meta, geometry, fields = read(naturalearth_lowres)

    assert meta["crs"] == "EPSG:4326"
    assert meta["geometry_type"] == "Polygon"
    assert meta["encoding"] == "UTF-8"
    assert meta["fields"].shape == (94,)

    assert meta["fields"].tolist() == [
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
    ]

    assert len(fields) == 94
    assert len(geometry) == len(fields[0])

    # quick test that WKB is a Polygon type
    assert geometry[0][:6] == b"\x01\x06\x00\x00\x00\x03"


def test_vsi_read_layers(naturalearth_modres_vsi):
    assert array_equal(
        list_layers(naturalearth_modres_vsi), [["ne_10m_admin_0_countries", "Polygon"]]
    )

    meta, geometry, fields = read(naturalearth_modres_vsi)
    assert geometry.shape == (255,)


def test_read_no_geometry(naturalearth_lowres):
    meta, geometry, fields = read(naturalearth_lowres, read_geometry=False)

    assert geometry is None


def test_read_columns(naturalearth_lowres):
    # read no columns or geometry
    meta, geometry, fields = read(naturalearth_lowres, columns=[], read_geometry=False)
    assert geometry is None
    assert len(fields) == 0
    array_equal(meta["fields"], np.empty(shape=(0, 4), dtype="object"))

    columns = ["NAME", "NAME_LONG"]
    meta, geometry, fields = read(
        naturalearth_lowres, columns=columns, read_geometry=False
    )
    array_equal(meta["fields"], columns)

    # Repeats should be dropped
    columns = ["NAME", "NAME_LONG", "NAME"]
    meta, geometry, fields = read(
        naturalearth_lowres, columns=columns, read_geometry=False
    )
    array_equal(meta["fields"], columns[:2])


def test_read_skip_features(naturalearth_lowres):
    expected_geometry, expected_fields = read(naturalearth_lowres)[1:]
    geometry, fields = read(naturalearth_lowres, skip_features=10)[1:]

    assert len(geometry) == len(expected_geometry) - 10
    assert len(fields[0]) == len(expected_fields[0]) - 10

    assert np.array_equal(geometry, expected_geometry[10:])
    # Last field has more variable data
    assert np.array_equal(fields[-1], expected_fields[-1][10:])


def test_read_max_features(naturalearth_lowres):
    expected_geometry, expected_fields = read(naturalearth_lowres)[1:]
    geometry, fields = read(naturalearth_lowres, max_features=2)[1:]

    assert len(geometry) == 2
    assert len(fields[0]) == 2

    assert np.array_equal(geometry, expected_geometry[:2])
    assert np.array_equal(fields[-1], expected_fields[-1][:2])


def test_read_info(naturalearth_lowres):
    meta = read_info(naturalearth_lowres)

    assert meta["crs"] == "EPSG:4326"
    assert meta["geometry_type"] == "Polygon"
    assert meta["encoding"] == "UTF-8"
    assert meta["fields"].shape == (94,)
    assert meta["features"] == 177


def test_write(tmpdir, naturalearth_lowres):
    meta, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.shp")
    write(filename, geometry, field_data, **meta)

    assert os.path.exists(filename)
    for ext in (".dbf", ".prj"):
        assert os.path.exists(filename.replace(".shp", ext))


def test_write_gpkg(tmpdir, naturalearth_lowres):
    meta, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.gpkg")
    write(filename, geometry, field_data, driver="GPKG", **meta)

    assert os.path.exists(filename)


def test_write_geojson(tmpdir, naturalearth_lowres):
    meta, geometry, field_data = read(naturalearth_lowres)

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


def test_write_geojsonseq(tmpdir, naturalearth_lowres):
    meta, geometry, field_data = read(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.json")
    write(filename, geometry, field_data, driver="GeoJSONSeq", **meta)

    assert os.path.exists(filename)

