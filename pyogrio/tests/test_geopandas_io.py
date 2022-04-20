import os

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import pytest

from pyogrio import list_layers
from pyogrio.errors import DataLayerError
from pyogrio.geopandas import read_dataframe, write_dataframe

try:
    import geopandas as gp
    from geopandas.testing import assert_geodataframe_equal

    has_geopandas = True
except ImportError:
    has_geopandas = False


pytestmark = pytest.mark.skipif(not has_geopandas, reason="GeoPandas not available")


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
    df = read_dataframe(naturalearth_lowres_vsi[1])
    assert len(df) == 177


def test_read_no_geometry(naturalearth_lowres):
    df = read_dataframe(naturalearth_lowres, read_geometry=False)
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df, gp.GeoDataFrame)


def test_read_force_2d(test_fgdb_vsi):
    with pytest.warns(
        UserWarning, match=r"Measured \(M\) geometry types are not supported"
    ):
        df = read_dataframe(test_fgdb_vsi, layer="test_lines", max_features=1)
        assert df.iloc[0].geometry.has_z

        df = read_dataframe(
            test_fgdb_vsi, layer="test_lines", force_2d=True, max_features=1
        )
        assert not df.iloc[0].geometry.has_z


@pytest.mark.filterwarnings("ignore: Measured")
def test_read_layer(test_fgdb_vsi):
    layers = list_layers(test_fgdb_vsi)
    # The first layer is read by default (NOTE: first layer has no features)
    df = read_dataframe(test_fgdb_vsi, read_geometry=False, max_features=1)
    df2 = read_dataframe(
        test_fgdb_vsi, layer=layers[0][0], read_geometry=False, max_features=1
    )
    assert_frame_equal(df, df2)

    # Reading a specific layer should return that layer.
    # Detected here by a known column.
    df = read_dataframe(
        test_fgdb_vsi, layer="test_lines", read_geometry=False, max_features=1
    )
    assert "RIVER_MILE" in df.columns


def test_read_layer_invalid(naturalearth_lowres):
    with pytest.raises(DataLayerError, match="Layer 'wrong' could not be opened"):
        read_dataframe(naturalearth_lowres, layer="wrong")


@pytest.mark.filterwarnings("ignore: Measured")
def test_read_datetime(test_fgdb_vsi):
    df = read_dataframe(test_fgdb_vsi, layer="test_lines", max_features=1)
    assert df.SURVEY_DAT.dtype.name == "datetime64[ns]"


def test_read_null_values(test_fgdb_vsi):
    df = read_dataframe(test_fgdb_vsi, read_geometry=False)

    # make sure that Null values are preserved
    assert df.SEGMENT_NAME.isnull().max() == True
    assert df.loc[df.SEGMENT_NAME.isnull()].SEGMENT_NAME.iloc[0] == None


def test_read_fid_as_index(naturalearth_lowres):
    kwargs = {"skip_features": 2, "max_features": 2}

    # default is to not set FIDs as index
    df = read_dataframe(naturalearth_lowres, **kwargs)
    assert_index_equal(df.index, pd.RangeIndex(0, 2))

    df = read_dataframe(naturalearth_lowres, fid_as_index=False, **kwargs)
    assert_index_equal(df.index, pd.RangeIndex(0, 2))

    df = read_dataframe(naturalearth_lowres, fid_as_index=True, **kwargs)
    assert_index_equal(df.index, pd.Index([2, 3], name="fid"))


@pytest.mark.filterwarnings("ignore: Layer")
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


@pytest.mark.parametrize("bbox", [(1,), (1, 2), (1, 2, 3)])
def test_read_bbox_invalid(naturalearth_lowres, bbox):
    with pytest.raises(ValueError, match="Invalid bbox"):
        read_dataframe(naturalearth_lowres, bbox=bbox)


def test_read_bbox(naturalearth_lowres):
    # should return no features
    with pytest.warns(UserWarning, match="does not have any features to read"):
        df = read_dataframe(naturalearth_lowres, bbox=(0, 0, 0.00001, 0.00001))
        assert len(df) == 0

    df = read_dataframe(naturalearth_lowres, bbox=(-140, 20, -100, 40))
    assert len(df) == 2
    assert np.array_equal(df.iso_a3, ["USA", "MEX"])


def test_read_fids(naturalearth_lowres):
    # ensure keyword is properly passed through
    fids = np.array([0, 10, 5], dtype=np.int64)
    df = read_dataframe(naturalearth_lowres, fids=fids, fid_as_index=True)
    assert len(df) == 3
    assert np.array_equal(fids, df.index.values)


def test_read_fids_force_2d(test_fgdb_vsi):
    with pytest.warns(
        UserWarning, match=r"Measured \(M\) geometry types are not supported"
    ):
        df = read_dataframe(test_fgdb_vsi, layer="test_lines", fids=[22])
        assert len(df) == 1
        assert df.iloc[0].geometry.has_z

        df = read_dataframe(test_fgdb_vsi, layer="test_lines", force_2d=True, fids=[22])
        assert len(df) == 1
        assert not df.iloc[0].geometry.has_z


@pytest.mark.filterwarnings("ignore: Layer")
def test_read_sql(naturalearth_lowres):
    # The geometry column cannot be specified when using the 
    # default OGRSQL dialect but is returned nonetheless, so 4 columns.
    sql = "SELECT iso_a3 AS iso_a3_renamed, name, pop_est FROM naturalearth_lowres"
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="OGRSQL")
    assert len(df.columns) == 4
    assert len(df) == 177

    # Columns and/or where filters are applied after the sql query, so alias 
    # needs to be specified instead of original column name.
    df = read_dataframe(
        naturalearth_lowres, 
        sql=sql, 
        sql_dialect="OGRSQL", 
        columns=["iso_a3_renamed", "name"],
        where="iso_a3_renamed IN ('CAN', 'USA', 'MEX')",
    )
    assert len(df.columns) == 3
    assert len(df) == 3
    assert len(set(df.iso_a3_renamed.unique()).difference(["CAN", "MEX", "USA"])) == 0

    # Add bbox as well.
    df = read_dataframe(
        naturalearth_lowres, 
        sql=sql, 
        sql_dialect="OGRSQL", 
        columns=["iso_a3_renamed", "name"],
        where="iso_a3_renamed IN ('CAN', 'USA', 'MEX')",
        bbox=(-140, 20, -100, 40)
    )
    assert len(df.columns) == 3
    assert len(df) == 2
    assert len(set(df.iso_a3_renamed.unique()).difference(["MEX", "USA"])) == 0

    # Should return single row.
    sql = "SELECT * FROM naturalearth_lowres WHERE iso_a3 = 'CAN'"
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="OGRSQL")
    assert len(df) == 1
    assert len(df.columns) == 6
    assert df.iloc[0].iso_a3 == "CAN"
    
    sql = """SELECT *  
               FROM naturalearth_lowres 
              WHERE iso_a3 IN ('CAN', 'USA', 'MEX')"""
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="OGRSQL")
    assert len(df.columns) == 6
    assert len(df) == 3
    assert len(set(df.iso_a3.unique()).difference(["CAN", "USA", "MEX"])) == 0
    
    sql = """SELECT *
               FROM naturalearth_lowres 
              WHERE iso_a3 IN ('CAN', 'USA', 'MEX')
              ORDER BY name"""
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="OGRSQL")
    assert len(df.columns) == 6
    assert len(df) == 3
    assert df.iso_a3.tolist() == ["CAN", "MEX", "USA"]

    sql = """SELECT *
               FROM naturalearth_lowres 
              WHERE iso_a3 IN ('CAN', 'MEX', 'USA')
              ORDER BY name"""
    df = read_dataframe(
            naturalearth_lowres, sql=sql, skip_features=1, max_features=1, sql_dialect="OGRSQL")
    assert len(df.columns) == 6
    assert len(df) == 1
    assert df.iso_a3.tolist() == ["MEX"]
    
    # Should return items within range.
    sql = """SELECT *
               FROM naturalearth_lowres 
              WHERE POP_EST >= 10000000 AND POP_EST < 100000000"""
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="OGRSQL")
    assert len(df) == 75
    assert len(df.columns) == 6
    assert df.pop_est.min() >= 10000000
    assert df.pop_est.max() < 100000000

    # Should match no items.
    sql = "SELECT * FROM naturalearth_lowres WHERE ISO_A3 = 'INVALID'"
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="OGRSQL")
    assert len(df) == 0

    sql = "SELECT * FROM naturalearth_lowres LIMIT 1"
    df = read_dataframe(naturalearth_lowres, sql=sql, max_features=3, sql_dialect="OGRSQL")
    assert len(df) == 1

    sql = "SELECT * FROM naturalearth_lowres LIMIT 1"
    with pytest.raises(ValueError, match="'skip_features' must be between 0 and 0"):
        df = read_dataframe(naturalearth_lowres, sql=sql, skip_features=1, sql_dialect="OGRSQL")

def test_read_sql_dialect_sqlite(naturalearth_lowres):
    # should return singular item
    sql = "SELECT * FROM naturalearth_lowres WHERE iso_a3 = 'CAN'"
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="SQLITE")
    assert len(df) == 1
    assert len(df.columns) == 6
    assert df.iloc[0].iso_a3 == "CAN"
    area_canada = df.iloc[0].geometry.area

    # use spatialite function (needs SQLITE dialect)
    sql = """SELECT ST_Buffer(geometry, 5) AS geometry, name, pop_est, iso_a3 
               FROM naturalearth_lowres 
              WHERE ISO_A3 = 'CAN'"""
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="SQLITE")
    assert len(df) == 1
    assert len(df.columns) == 4
    assert df.iloc[0].geometry.area > area_canada


def test_read_sql_invalid(naturalearth_lowres):
    with pytest.raises(Exception, match="SQL Expression Parsing Error"):
        read_dataframe(naturalearth_lowres, sql="invalid")
    with pytest.raises(ValueError, match="'sql' paramater cannot be combined with 'layer'"):
        read_dataframe(naturalearth_lowres, sql="whatever", layer="invalid")
        

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
            df,
            expected,
            check_less_precise=is_json,
            check_index_type=False,
            check_dtype=not is_json,
        )


@pytest.mark.parametrize(
    "driver,ext", [("ESRI Shapefile", "shp"), ("GeoJSON", "geojson"), ("GPKG", "gpkg")]
)
def test_write_empty_dataframe(tmpdir, driver, ext):
    expected = gp.GeoDataFrame(geometry=[], crs=4326)

    filename = os.path.join(str(tmpdir), f"test.{ext}")
    write_dataframe(expected, filename, driver=driver)

    assert os.path.exists(filename)

    df = read_dataframe(filename)

    assert_geodataframe_equal(df, expected)


def test_write_dataframe_gdalparams(tmpdir, naturalearth_lowres):
    original_df = read_dataframe(naturalearth_lowres)
    
    test_noindex_filename = os.path.join(str(tmpdir), f"test_gdalparams_noindex.shp")
    write_dataframe(original_df, test_noindex_filename, SPATIAL_INDEX="NO")
    assert os.path.exists(test_noindex_filename) is True
    test_noindex_index_filename = os.path.join(str(tmpdir), f"test_gdalparams_noindex.qix")
    assert os.path.exists(test_noindex_index_filename) is False
    
    test_withindex_filename = os.path.join(str(tmpdir), f"test_gdalparams_withindex.shp")
    write_dataframe(original_df, test_withindex_filename, SPATIAL_INDEX="YES")
    assert os.path.exists(test_withindex_filename) is True
    test_withindex_index_filename = os.path.join(str(tmpdir), f"test_gdalparams_withindex.qix")
    assert os.path.exists(test_withindex_index_filename) is True


@pytest.mark.filterwarnings(
    "ignore: You will likely lose important projection information"
)
def test_custom_crs_io(tmpdir, naturalearth_lowres):
    df = read_dataframe(naturalearth_lowres)
    # project Belgium to a custom Albers Equal Area projection
    expected = df.loc[df.name == "Belgium"].to_crs(
        "+proj=aea +lat_1=49.5 +lat_2=51.5 +lon_0=4.3"
    )
    filename = os.path.join(str(tmpdir), "test.shp")
    write_dataframe(expected, filename)

    assert os.path.exists(filename)

    df = read_dataframe(filename)

    crs = df.crs.to_dict()
    assert crs["lat_1"] == 49.5
    assert crs["lat_2"] == 51.5
    assert crs["lon_0"] == 4.3
    assert df.crs.equals(expected.crs)

