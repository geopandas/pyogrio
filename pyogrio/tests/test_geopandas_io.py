import contextlib
from datetime import datetime
import os
import numpy as np
import pytest

from pyogrio import list_layers, read_info, __gdal_version__
from pyogrio.errors import DataLayerError, DataSourceError, FeatureError, GeometryError
from pyogrio.geopandas import read_dataframe, write_dataframe, PANDAS_GE_20
from pyogrio.raw import (
    DRIVERS_NO_MIXED_DIMENSIONS,
    DRIVERS_NO_MIXED_SINGLE_MULTI,
)
from pyogrio.tests.conftest import (
    ALL_EXTS,
    DRIVERS,
    requires_arrow_api,
    requires_gdal_geos,
)
from pyogrio._compat import PANDAS_GE_15

try:
    import pandas as pd
    from pandas.testing import (
        assert_frame_equal,
        assert_index_equal,
        assert_series_equal,
    )

    import geopandas as gp
    from geopandas.array import from_wkt
    from geopandas.testing import assert_geodataframe_equal

    import shapely  # if geopandas is present, shapely is expected to be present
    from shapely.geometry import Point

except ImportError:
    pass


pytest.importorskip("geopandas")


@pytest.fixture(
    scope="session",
    params=[
        False,
        pytest.param(True, marks=requires_arrow_api),
    ],
)
def use_arrow(request):
    return request.param


def spatialite_available(path):
    try:
        _ = read_dataframe(
            path, sql="select spatialite_version();", sql_dialect="SQLITE"
        )
        return True
    except Exception:
        return False


def test_read_dataframe(naturalearth_lowres_all_ext):
    df = read_dataframe(naturalearth_lowres_all_ext)

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


def test_read_dataframe_vsi(naturalearth_lowres_vsi):
    df = read_dataframe(naturalearth_lowres_vsi[1])
    assert len(df) == 177


@pytest.mark.parametrize(
    "columns, fid_as_index, exp_len", [(None, False, 2), ([], True, 2), ([], False, 0)]
)
def test_read_layer_without_geometry(
    test_fgdb_vsi, columns, fid_as_index, use_arrow, exp_len
):
    result = read_dataframe(
        test_fgdb_vsi,
        layer="basetable",
        columns=columns,
        fid_as_index=fid_as_index,
        use_arrow=use_arrow,
    )
    assert type(result) is pd.DataFrame
    assert len(result) == exp_len


@pytest.mark.parametrize(
    "naturalearth_lowres, expected_ext",
    [(".gpkg", ".gpkg"), (".shp", ".shp")],
    indirect=["naturalearth_lowres"],
)
def test_fixture_naturalearth_lowres(naturalearth_lowres, expected_ext):
    # Test the fixture with "indirect" parameter
    assert naturalearth_lowres.suffix == expected_ext
    df = read_dataframe(naturalearth_lowres)
    assert len(df) == 177


def test_read_no_geometry(naturalearth_lowres_all_ext, use_arrow):
    df = read_dataframe(
        naturalearth_lowres_all_ext, use_arrow=use_arrow, read_geometry=False
    )
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df, gp.GeoDataFrame)


def test_read_no_geometry_no_columns_no_fids(naturalearth_lowres, use_arrow):
    with pytest.raises(
        ValueError,
        match=(
            "at least one of read_geometry or return_fids must be True or columns must "
            "be None or non-empty"
        ),
    ):
        _ = read_dataframe(
            naturalearth_lowres,
            columns=[],
            read_geometry=False,
            fid_as_index=False,
            use_arrow=use_arrow,
        )


def test_read_force_2d(test_fgdb_vsi, use_arrow):
    with pytest.warns(
        UserWarning, match=r"Measured \(M\) geometry types are not supported"
    ):
        df = read_dataframe(test_fgdb_vsi, layer="test_lines", max_features=1)
        assert df.iloc[0].geometry.has_z

        df = read_dataframe(
            test_fgdb_vsi,
            layer="test_lines",
            force_2d=True,
            max_features=1,
            use_arrow=use_arrow,
        )
        assert not df.iloc[0].geometry.has_z


@pytest.mark.filterwarnings("ignore: Measured")
def test_read_layer(test_fgdb_vsi, use_arrow):
    layers = list_layers(test_fgdb_vsi)
    kwargs = {"use_arrow": use_arrow, "read_geometry": False, "max_features": 1}

    # The first layer is read by default (NOTE: first layer has no features)
    df = read_dataframe(test_fgdb_vsi, **kwargs)
    df2 = read_dataframe(test_fgdb_vsi, layer=layers[0][0], **kwargs)
    assert_frame_equal(df, df2)

    # Reading a specific layer should return that layer.
    # Detected here by a known column.
    df = read_dataframe(test_fgdb_vsi, layer="test_lines", **kwargs)
    assert "RIVER_MILE" in df.columns


def test_read_layer_invalid(naturalearth_lowres_all_ext, use_arrow):
    with pytest.raises(DataLayerError, match="Layer 'wrong' could not be opened"):
        read_dataframe(naturalearth_lowres_all_ext, layer="wrong", use_arrow=use_arrow)


@pytest.mark.filterwarnings("ignore: Measured")
def test_read_datetime(test_fgdb_vsi, use_arrow):
    df = read_dataframe(
        test_fgdb_vsi, layer="test_lines", use_arrow=use_arrow, max_features=1
    )
    if PANDAS_GE_20:
        # starting with pandas 2.0, it preserves the passed datetime resolution
        assert df.SURVEY_DAT.dtype.name == "datetime64[ms]"
    else:
        assert df.SURVEY_DAT.dtype.name == "datetime64[ns]"


def test_read_datetime_tz(test_datetime_tz, tmp_path):
    df = read_dataframe(test_datetime_tz)
    raw_expected = ["2020-01-01T09:00:00.123-05:00", "2020-01-01T10:00:00-05:00"]

    if PANDAS_GE_20:
        expected = pd.to_datetime(raw_expected, format="ISO8601").as_unit("ms")
    else:
        expected = pd.to_datetime(raw_expected)
    expected = pd.Series(expected, name="datetime_col")
    assert_series_equal(df.datetime_col, expected)
    # test write and read round trips
    fpath = tmp_path / "test.gpkg"
    write_dataframe(df, fpath)
    df_read = read_dataframe(fpath)
    assert_series_equal(df_read.datetime_col, expected)


def test_write_datetime_mixed_offset(tmp_path):
    # Australian Summer Time AEDT (GMT+11), Standard Time AEST (GMT+10)
    dates = ["2023-01-01 11:00:01.111", "2023-06-01 10:00:01.111"]
    naive_col = pd.Series(pd.to_datetime(dates), name="dates")
    localised_col = naive_col.dt.tz_localize("Australia/Sydney")
    utc_col = localised_col.dt.tz_convert("UTC")
    if PANDAS_GE_20:
        utc_col = utc_col.dt.as_unit("ms")

    df = gp.GeoDataFrame(
        {"dates": localised_col, "geometry": [Point(1, 1), Point(1, 1)]},
        crs="EPSG:4326",
    )
    fpath = tmp_path / "test.gpkg"
    write_dataframe(df, fpath)
    result = read_dataframe(fpath)
    # GDAL tz only encodes offsets, not timezones
    # check multiple offsets are read as utc datetime instead of string values
    assert_series_equal(result["dates"], utc_col)


def test_read_write_datetime_tz_with_nulls(tmp_path):
    dates_raw = ["2020-01-01T09:00:00.123-05:00", "2020-01-01T10:00:00-05:00", pd.NaT]
    if PANDAS_GE_20:
        dates = pd.to_datetime(dates_raw, format="ISO8601").as_unit("ms")
    else:
        dates = pd.to_datetime(dates_raw)
    df = gp.GeoDataFrame(
        {"dates": dates, "geometry": [Point(1, 1), Point(1, 1), Point(1, 1)]},
        crs="EPSG:4326",
    )
    fpath = tmp_path / "test.gpkg"
    write_dataframe(df, fpath)
    result = read_dataframe(fpath)
    assert_geodataframe_equal(df, result)


def test_read_null_values(test_fgdb_vsi, use_arrow):
    df = read_dataframe(test_fgdb_vsi, use_arrow=use_arrow, read_geometry=False)

    # make sure that Null values are preserved
    assert df.SEGMENT_NAME.isnull().max()
    assert df.loc[df.SEGMENT_NAME.isnull()].SEGMENT_NAME.iloc[0] is None


def test_read_fid_as_index(naturalearth_lowres_all_ext, use_arrow):
    kwargs = {"use_arrow": use_arrow, "skip_features": 2, "max_features": 2}

    # default is to not set FIDs as index
    df = read_dataframe(naturalearth_lowres_all_ext, **kwargs)
    assert_index_equal(df.index, pd.RangeIndex(0, 2))

    df = read_dataframe(naturalearth_lowres_all_ext, fid_as_index=False, **kwargs)
    assert_index_equal(df.index, pd.RangeIndex(0, 2))

    df = read_dataframe(
        naturalearth_lowres_all_ext,
        fid_as_index=True,
        **kwargs,
    )
    if naturalearth_lowres_all_ext.suffix in [".gpkg"]:
        # File format where fid starts at 1
        assert_index_equal(df.index, pd.Index([3, 4], name="fid"))
    else:
        # File format where fid starts at 0
        assert_index_equal(df.index, pd.Index([2, 3], name="fid"))


def test_read_fid_as_index_only(naturalearth_lowres, use_arrow):
    df = read_dataframe(
        naturalearth_lowres,
        columns=[],
        read_geometry=False,
        fid_as_index=True,
        use_arrow=use_arrow,
    )
    assert df is not None
    assert len(df) == 177
    assert len(df.columns) == 0


def test_read_where(naturalearth_lowres_all_ext, use_arrow):
    # empty filter should return full set of records
    df = read_dataframe(naturalearth_lowres_all_ext, use_arrow=use_arrow, where="")
    assert len(df) == 177

    # should return singular item
    df = read_dataframe(
        naturalearth_lowres_all_ext, use_arrow=use_arrow, where="iso_a3 = 'CAN'"
    )
    assert len(df) == 1
    assert df.iloc[0].iso_a3 == "CAN"

    df = read_dataframe(
        naturalearth_lowres_all_ext,
        use_arrow=use_arrow,
        where="iso_a3 IN ('CAN', 'USA', 'MEX')",
    )
    assert len(df) == 3
    assert len(set(df.iso_a3.unique()).difference(["CAN", "USA", "MEX"])) == 0

    # should return items within range
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        use_arrow=use_arrow,
        where="POP_EST >= 10000000 AND POP_EST < 100000000",
    )
    assert len(df) == 75
    assert df.pop_est.min() >= 10000000
    assert df.pop_est.max() < 100000000

    # should match no items
    df = read_dataframe(
        naturalearth_lowres_all_ext, use_arrow=use_arrow, where="ISO_A3 = 'INVALID'"
    )
    assert len(df) == 0


def test_read_where_invalid(request, naturalearth_lowres_all_ext, use_arrow):
    if use_arrow and naturalearth_lowres_all_ext.suffix == ".gpkg":
        # https://github.com/OSGeo/gdal/issues/8492
        request.node.add_marker(pytest.mark.xfail(reason="GDAL doesn't error for GPGK"))
    with pytest.raises(ValueError, match="Invalid SQL"):
        read_dataframe(
            naturalearth_lowres_all_ext, use_arrow=use_arrow, where="invalid"
        )


@pytest.mark.parametrize("bbox", [(1,), (1, 2), (1, 2, 3)])
def test_read_bbox_invalid(naturalearth_lowres_all_ext, bbox, use_arrow):
    with pytest.raises(ValueError, match="Invalid bbox"):
        read_dataframe(naturalearth_lowres_all_ext, use_arrow=use_arrow, bbox=bbox)


@pytest.mark.parametrize(
    "bbox,expected",
    [
        ((0, 0, 0.00001, 0.00001), []),
        ((-85, 8, -80, 10), ["PAN", "CRI"]),
        ((-104, 54, -105, 55), ["CAN"]),
    ],
)
def test_read_bbox(naturalearth_lowres_all_ext, use_arrow, bbox, expected):
    if (
        use_arrow
        and __gdal_version__ < (3, 8, 0)
        and os.path.splitext(naturalearth_lowres_all_ext)[1] == ".gpkg"
    ):
        pytest.xfail(reason="GDAL bug: https://github.com/OSGeo/gdal/issues/8347")

    df = read_dataframe(naturalearth_lowres_all_ext, use_arrow=use_arrow, bbox=bbox)

    assert np.array_equal(df.iso_a3, expected)


def test_read_bbox_sql(naturalearth_lowres_all_ext, use_arrow):
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        use_arrow=use_arrow,
        bbox=(-180, 50, -100, 90),
        sql="SELECT * from naturalearth_lowres where iso_a3 not in ('USA', 'RUS')",
    )
    assert len(df) == 1
    assert np.array_equal(df.iso_a3, ["CAN"])


def test_read_bbox_where(naturalearth_lowres_all_ext, use_arrow):
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        use_arrow=use_arrow,
        bbox=(-180, 50, -100, 90),
        where="iso_a3 not in ('USA', 'RUS')",
    )
    assert len(df) == 1
    assert np.array_equal(df.iso_a3, ["CAN"])


@pytest.mark.parametrize(
    "mask",
    [
        {"type": "Point", "coordinates": [0, 0]},
        '{"type": "Point", "coordinates": [0, 0]}',
        "invalid",
    ],
)
def test_read_mask_invalid(naturalearth_lowres, use_arrow, mask):
    with pytest.raises(ValueError, match="'mask' parameter must be a Shapely geometry"):
        read_dataframe(naturalearth_lowres, use_arrow=use_arrow, mask=mask)


def test_read_bbox_mask_invalid(naturalearth_lowres, use_arrow):
    with pytest.raises(ValueError, match="cannot set both 'bbox' and 'mask'"):
        read_dataframe(
            naturalearth_lowres,
            use_arrow=use_arrow,
            bbox=(-85, 8, -80, 10),
            mask=shapely.Point(-105, 55),
        )


@pytest.mark.parametrize(
    "mask,expected",
    [
        (shapely.Point(-105, 55), ["CAN"]),
        (shapely.box(-85, 8, -80, 10), ["PAN", "CRI"]),
        (
            shapely.Polygon(
                (
                    [6.101929483362767, 50.97085041206964],
                    [5.773001596839322, 50.90661120482673],
                    [5.593156133704326, 50.642648747710325],
                    [6.059271089606312, 50.686051894002475],
                    [6.374064065737485, 50.851481340346965],
                    [6.101929483362767, 50.97085041206964],
                )
            ),
            ["DEU", "BEL", "NLD"],
        ),
        (
            shapely.GeometryCollection(
                [shapely.Point(-7.7, 53), shapely.box(-85, 8, -80, 10)]
            ),
            ["PAN", "CRI", "IRL"],
        ),
    ],
)
def test_read_mask(
    naturalearth_lowres_all_ext,
    use_arrow,
    mask,
    expected,
):
    if (
        use_arrow
        and __gdal_version__ < (3, 8, 0)
        and os.path.splitext(naturalearth_lowres_all_ext)[1] == ".gpkg"
    ):
        pytest.xfail(reason="GDAL bug: https://github.com/OSGeo/gdal/issues/8347")

    df = read_dataframe(naturalearth_lowres_all_ext, use_arrow=use_arrow, mask=mask)

    assert len(df) == len(expected)
    assert np.array_equal(df.iso_a3, expected)


def test_read_mask_sql(naturalearth_lowres_all_ext, use_arrow):
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        use_arrow=use_arrow,
        mask=shapely.box(-180, 50, -100, 90),
        sql="SELECT * from naturalearth_lowres where iso_a3 not in ('USA', 'RUS')",
    )
    assert len(df) == 1
    assert np.array_equal(df.iso_a3, ["CAN"])


def test_read_mask_where(naturalearth_lowres_all_ext, use_arrow):
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        use_arrow=use_arrow,
        mask=shapely.box(-180, 50, -100, 90),
        where="iso_a3 not in ('USA', 'RUS')",
    )
    assert len(df) == 1
    assert np.array_equal(df.iso_a3, ["CAN"])


def test_read_fids(naturalearth_lowres_all_ext):
    # ensure keyword is properly passed through
    fids = np.array([1, 10, 5], dtype=np.int64)
    df = read_dataframe(naturalearth_lowres_all_ext, fids=fids, fid_as_index=True)
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


@pytest.mark.parametrize("skip_features", [10, 200])
def test_read_skip_features(naturalearth_lowres_all_ext, use_arrow, skip_features):
    ext = naturalearth_lowres_all_ext.suffix
    expected = (
        read_dataframe(naturalearth_lowres_all_ext)
        .iloc[skip_features:]
        .reset_index(drop=True)
    )

    df = read_dataframe(
        naturalearth_lowres_all_ext, skip_features=skip_features, use_arrow=use_arrow
    )
    assert len(df) == len(expected)

    # Coordinates are not precisely equal when written to JSON
    # dtypes do not necessarily round-trip precisely through JSON
    is_json = ext in [".geojson", ".geojsonl"]
    # In .geojsonl the vertices are reordered, so normalize
    is_jsons = ext == ".geojsonl"

    assert_geodataframe_equal(
        df,
        expected,
        check_less_precise=is_json,
        check_index_type=False,
        check_dtype=not is_json,
        normalize=is_jsons,
    )


def test_read_negative_skip_features(naturalearth_lowres, use_arrow):
    with pytest.raises(ValueError, match="'skip_features' must be >= 0"):
        read_dataframe(naturalearth_lowres, skip_features=-1, use_arrow=use_arrow)


@pytest.mark.parametrize("max_features", [10, 100])
def test_read_max_features(naturalearth_lowres_all_ext, use_arrow, max_features):
    ext = naturalearth_lowres_all_ext.suffix
    expected = read_dataframe(naturalearth_lowres_all_ext).iloc[:max_features]
    df = read_dataframe(
        naturalearth_lowres_all_ext, max_features=max_features, use_arrow=use_arrow
    )

    assert len(df) == len(expected)

    # Coordinates are not precisely equal when written to JSON
    # dtypes do not necessarily round-trip precisely through JSON
    is_json = ext in [".geojson", ".geojsonl"]
    # In .geojsonl the vertices are reordered, so normalize
    is_jsons = ext == ".geojsonl"

    assert_geodataframe_equal(
        df,
        expected,
        check_less_precise=is_json,
        check_index_type=False,
        check_dtype=not is_json,
        normalize=is_jsons,
    )


def test_read_negative_max_features(naturalearth_lowres, use_arrow):
    with pytest.raises(ValueError, match="'max_features' must be >= 0"):
        read_dataframe(naturalearth_lowres, max_features=-1, use_arrow=use_arrow)


def test_read_non_existent_file(use_arrow):
    # ensure consistent error type / message from GDAL
    with pytest.raises(DataSourceError, match="No such file or directory"):
        read_dataframe("non-existent.shp", use_arrow=use_arrow)

    with pytest.raises(DataSourceError, match="does not exist in the file system"):
        read_dataframe("/vsizip/non-existent.zip", use_arrow=use_arrow)

    with pytest.raises(DataSourceError, match="does not exist in the file system"):
        read_dataframe("zip:///non-existent.zip", use_arrow=use_arrow)


def test_read_sql(naturalearth_lowres_all_ext, use_arrow):
    # The geometry column cannot be specified when using the
    # default OGRSQL dialect but is returned nonetheless, so 4 columns.
    sql = "SELECT iso_a3 AS iso_a3_renamed, name, pop_est FROM naturalearth_lowres"
    df = read_dataframe(naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL")
    assert len(df.columns) == 4
    assert len(df) == 177

    # Should return single row
    sql = "SELECT * FROM naturalearth_lowres WHERE iso_a3 = 'CAN'"
    df = read_dataframe(naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL")
    assert len(df) == 1
    assert len(df.columns) == 6
    assert df.iloc[0].iso_a3 == "CAN"

    sql = """SELECT *
               FROM naturalearth_lowres
              WHERE iso_a3 IN ('CAN', 'USA', 'MEX')"""
    df = read_dataframe(naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL")
    assert len(df.columns) == 6
    assert len(df) == 3
    assert df.iso_a3.tolist() == ["CAN", "USA", "MEX"]

    sql = """SELECT *
               FROM naturalearth_lowres
              WHERE iso_a3 IN ('CAN', 'USA', 'MEX')
              ORDER BY name"""
    df = read_dataframe(naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL")
    assert len(df.columns) == 6
    assert len(df) == 3
    assert df.iso_a3.tolist() == ["CAN", "MEX", "USA"]

    # Should return items within range.
    sql = """SELECT *
               FROM naturalearth_lowres
              WHERE POP_EST >= 10000000 AND POP_EST < 100000000"""
    df = read_dataframe(naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL")
    assert len(df) == 75
    assert len(df.columns) == 6
    assert df.pop_est.min() >= 10000000
    assert df.pop_est.max() < 100000000

    # Should match no items.
    sql = "SELECT * FROM naturalearth_lowres WHERE ISO_A3 = 'INVALID'"
    df = read_dataframe(naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL")
    assert len(df) == 0


def test_read_sql_invalid(naturalearth_lowres_all_ext):
    if naturalearth_lowres_all_ext.suffix == ".gpkg":
        with pytest.raises(Exception, match="In ExecuteSQL().*"):
            read_dataframe(naturalearth_lowres_all_ext, sql="invalid")
    else:
        with pytest.raises(Exception, match="SQL Expression Parsing Error"):
            read_dataframe(naturalearth_lowres_all_ext, sql="invalid")

    with pytest.raises(
        ValueError, match="'sql' paramater cannot be combined with 'layer'"
    ):
        read_dataframe(naturalearth_lowres_all_ext, sql="whatever", layer="invalid")


def test_read_sql_columns_where(naturalearth_lowres_all_ext):
    sql = "SELECT iso_a3 AS iso_a3_renamed, name, pop_est FROM naturalearth_lowres"
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        sql=sql,
        sql_dialect="OGRSQL",
        columns=["iso_a3_renamed", "name"],
        where="iso_a3_renamed IN ('CAN', 'USA', 'MEX')",
    )
    assert len(df.columns) == 3
    assert len(df) == 3
    assert df.iso_a3_renamed.tolist() == ["CAN", "USA", "MEX"]


def test_read_sql_columns_where_bbox(naturalearth_lowres_all_ext):
    sql = "SELECT iso_a3 AS iso_a3_renamed, name, pop_est FROM naturalearth_lowres"
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        sql=sql,
        sql_dialect="OGRSQL",
        columns=["iso_a3_renamed", "name"],
        where="iso_a3_renamed IN ('CRI', 'PAN')",
        bbox=(-85, 8, -80, 10),
    )
    assert len(df.columns) == 3
    assert len(df) == 2
    assert df.iso_a3_renamed.tolist() == ["PAN", "CRI"]


def test_read_sql_skip_max(naturalearth_lowres_all_ext):
    sql = """SELECT *
               FROM naturalearth_lowres
              WHERE iso_a3 IN ('CAN', 'MEX', 'USA')
              ORDER BY name"""
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        sql=sql,
        skip_features=1,
        max_features=1,
        sql_dialect="OGRSQL",
    )
    assert len(df.columns) == 6
    assert len(df) == 1
    assert df.iso_a3.tolist() == ["MEX"]

    sql = "SELECT * FROM naturalearth_lowres LIMIT 1"
    df = read_dataframe(
        naturalearth_lowres_all_ext, sql=sql, max_features=3, sql_dialect="OGRSQL"
    )
    assert len(df) == 1

    sql = "SELECT * FROM naturalearth_lowres LIMIT 1"
    df = read_dataframe(
        naturalearth_lowres_all_ext, sql=sql, skip_features=1, sql_dialect="OGRSQL"
    )
    assert len(df) == 0


@requires_gdal_geos
@pytest.mark.parametrize(
    "naturalearth_lowres",
    [ext for ext in ALL_EXTS if ext != ".gpkg"],
    indirect=["naturalearth_lowres"],
)
def test_read_sql_dialect_sqlite_nogpkg(naturalearth_lowres):
    # Should return singular item
    sql = "SELECT * FROM naturalearth_lowres WHERE iso_a3 = 'CAN'"
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="SQLITE")
    assert len(df) == 1
    assert len(df.columns) == 6
    assert df.iloc[0].iso_a3 == "CAN"
    area_canada = df.iloc[0].geometry.area

    # Use spatialite function
    sql = """SELECT ST_Buffer(geometry, 5) AS geometry, name, pop_est, iso_a3
               FROM naturalearth_lowres
              WHERE ISO_A3 = 'CAN'"""
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="SQLITE")
    assert len(df) == 1
    assert len(df.columns) == 4
    assert df.iloc[0].geometry.area > area_canada


@requires_gdal_geos
@pytest.mark.parametrize(
    "naturalearth_lowres", [".gpkg"], indirect=["naturalearth_lowres"]
)
def test_read_sql_dialect_sqlite_gpkg(naturalearth_lowres):
    # "INDIRECT_SQL" prohibits GDAL from passing the SQL statement to sqlite.
    # Because the statement is processed within GDAL it is possible to use
    # spatialite functions even if sqlite isn't built with spatialite support.
    sql = "SELECT * FROM naturalearth_lowres WHERE iso_a3 = 'CAN'"
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="INDIRECT_SQLITE")
    assert len(df) == 1
    assert len(df.columns) == 6
    assert df.iloc[0].iso_a3 == "CAN"
    area_canada = df.iloc[0].geometry.area

    # Use spatialite function
    sql = """SELECT ST_Buffer(geom, 5) AS geometry, name, pop_est, iso_a3
               FROM naturalearth_lowres
              WHERE ISO_A3 = 'CAN'"""
    df = read_dataframe(naturalearth_lowres, sql=sql, sql_dialect="INDIRECT_SQLITE")
    assert len(df) == 1
    assert len(df.columns) == 4
    assert df.iloc[0].geometry.area > area_canada


@pytest.mark.parametrize("ext", ALL_EXTS)
def test_write_dataframe(tmp_path, naturalearth_lowres, ext):
    input_gdf = read_dataframe(naturalearth_lowres)
    output_path = tmp_path / f"test{ext}"

    if ext == ".fgb":
        # For .fgb, spatial_index=False to avoid the rows being reordered
        write_dataframe(input_gdf, output_path, spatial_index=False)
    else:
        write_dataframe(input_gdf, output_path)

    assert output_path.exists()
    result_gdf = read_dataframe(output_path)

    geometry_types = result_gdf.geometry.type.unique()
    if DRIVERS[ext] in DRIVERS_NO_MIXED_SINGLE_MULTI:
        assert geometry_types == ["MultiPolygon"]
    else:
        assert set(geometry_types) == set(["MultiPolygon", "Polygon"])

    # Coordinates are not precisely equal when written to JSON
    # dtypes do not necessarily round-trip precisely through JSON
    is_json = ext in [".geojson", ".geojsonl"]
    # In .geojsonl the vertices are reordered, so normalize
    is_jsons = ext == ".geojsonl"

    assert_geodataframe_equal(
        result_gdf,
        input_gdf,
        check_less_precise=is_json,
        check_index_type=False,
        check_dtype=not is_json,
        normalize=is_jsons,
    )


@pytest.mark.filterwarnings("ignore:.*No SRS set on layer.*")
@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS + [".xlsx"] if ext != ".fgb"])
def test_write_dataframe_no_geom(tmp_path, naturalearth_lowres, ext):
    """Test writing a dataframe without a geometry column.

    FlatGeobuf (.fgb) doesn't seem to support this, and just writes an empty file.
    """
    # Prepare test data
    input_df = read_dataframe(naturalearth_lowres, read_geometry=False)
    output_path = tmp_path / f"test{ext}"

    # A shapefile without geometry column results in only a .dbf file.
    if ext == ".shp":
        output_path = output_path.with_suffix(".dbf")

    # Determine driver
    driver = DRIVERS[ext] if ext != ".xlsx" else "XLSX"

    write_dataframe(input_df, output_path, driver=driver)

    assert output_path.exists()
    result_df = read_dataframe(output_path)

    assert isinstance(result_df, pd.DataFrame)

    # some dtypes do not round-trip precisely through these file types
    check_dtype = ext not in [".geojson", ".geojsonl", ".xlsx"]

    if ext in [".gpkg", ".shp", ".xlsx"]:
        # These file types return a DataFrame when read.
        assert not isinstance(result_df, gp.GeoDataFrame)
        pd.testing.assert_frame_equal(
            result_df, input_df, check_index_type=False, check_dtype=check_dtype
        )
    else:
        # These file types return a GeoDataFrame with None Geometries when read.
        input_none_geom_gdf = gp.GeoDataFrame(
            input_df, geometry=np.repeat(None, len(input_df)), crs=4326
        )
        assert_geodataframe_equal(
            result_df,
            input_none_geom_gdf,
            check_index_type=False,
            check_dtype=check_dtype,
        )


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext not in ".geojsonl"])
def test_write_empty_dataframe(tmp_path, ext):
    expected = gp.GeoDataFrame(geometry=[], crs=4326)

    filename = tmp_path / f"test{ext}"
    write_dataframe(expected, filename)

    assert filename.exists()
    df = read_dataframe(filename)
    assert_geodataframe_equal(df, expected)


@pytest.mark.parametrize("ext", [".geojsonl", ".geojsons"])
def test_write_read_empty_dataframe_unsupported(tmp_path, ext):
    # Writing empty dataframe to .geojsons or .geojsonl results logically in a 0 byte
    # file, but gdal isn't able to read those again at the time of writing.
    # Issue logged here: https://github.com/geopandas/pyogrio/issues/94
    expected = gp.GeoDataFrame(geometry=[], crs=4326)

    filename = tmp_path / f"test{ext}"
    write_dataframe(expected, filename)

    assert filename.exists()
    with pytest.raises(
        Exception, match=".* not recognized as a supported file format."
    ):
        _ = read_dataframe(filename)


def test_write_dataframe_gpkg_multiple_layers(tmp_path, naturalearth_lowres):
    input_gdf = read_dataframe(naturalearth_lowres)
    output_path = tmp_path / "test.gpkg"

    write_dataframe(input_gdf, output_path, layer="first", promote_to_multi=True)

    assert os.path.exists(output_path)
    assert np.array_equal(list_layers(output_path), [["first", "MultiPolygon"]])

    write_dataframe(input_gdf, output_path, layer="second", promote_to_multi=True)
    assert np.array_equal(
        list_layers(output_path),
        [["first", "MultiPolygon"], ["second", "MultiPolygon"]],
    )


@pytest.mark.parametrize("ext", ALL_EXTS)
def test_write_dataframe_append(tmp_path, naturalearth_lowres, ext):
    if ext == ".fgb" and __gdal_version__ <= (3, 5, 0):
        pytest.skip("Append to FlatGeobuf fails for GDAL <= 3.5.0")

    if ext in (".geojsonl", ".geojsons") and __gdal_version__ <= (3, 6, 0):
        pytest.skip("Append to GeoJSONSeq only available for GDAL >= 3.6.0")

    input_gdf = read_dataframe(naturalearth_lowres)
    output_path = tmp_path / f"test{ext}"

    write_dataframe(input_gdf, output_path)

    assert os.path.exists(output_path)
    assert len(read_dataframe(output_path)) == 177

    write_dataframe(input_gdf, output_path, append=True)
    assert len(read_dataframe(output_path)) == 354


@pytest.mark.parametrize("spatial_index", [False, True])
def test_write_dataframe_gdal_options(tmp_path, naturalearth_lowres, spatial_index):
    df = read_dataframe(naturalearth_lowres)

    outfilename1 = tmp_path / "test1.shp"
    write_dataframe(df, outfilename1, SPATIAL_INDEX="YES" if spatial_index else "NO")
    assert outfilename1.exists() is True
    index_filename1 = tmp_path / "test1.qix"
    assert index_filename1.exists() is spatial_index

    # using explicit layer_options instead
    outfilename2 = tmp_path / "test2.shp"
    write_dataframe(df, outfilename2, layer_options=dict(spatial_index=spatial_index))
    assert outfilename2.exists() is True
    index_filename2 = tmp_path / "test2.qix"
    assert index_filename2.exists() is spatial_index


def test_write_dataframe_gdal_options_unknown(tmp_path, naturalearth_lowres):
    df = read_dataframe(naturalearth_lowres)

    # geojson has no spatial index, so passing keyword should raise
    outfilename = tmp_path / "test.geojson"
    with pytest.raises(ValueError, match="unrecognized option 'SPATIAL_INDEX'"):
        write_dataframe(df, outfilename, spatial_index=True)


def _get_gpkg_table_names(path):
    import sqlite3

    con = sqlite3.connect(path)
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    result = cursor.fetchall()
    return [res[0] for res in result]


def test_write_dataframe_gdal_options_dataset(tmp_path, naturalearth_lowres):
    df = read_dataframe(naturalearth_lowres)

    test_default_filename = tmp_path / "test_default.gpkg"
    write_dataframe(df, test_default_filename)
    assert "gpkg_ogr_contents" in _get_gpkg_table_names(test_default_filename)

    test_no_contents_filename = tmp_path / "test_no_contents.gpkg"
    write_dataframe(df, test_default_filename, ADD_GPKG_OGR_CONTENTS="NO")
    assert "gpkg_ogr_contents" not in _get_gpkg_table_names(test_no_contents_filename)

    test_no_contents_filename2 = tmp_path / "test_no_contents2.gpkg"
    write_dataframe(
        df,
        test_no_contents_filename2,
        dataset_options=dict(add_gpkg_ogr_contents=False),
    )
    assert "gpkg_ogr_contents" not in _get_gpkg_table_names(test_no_contents_filename2)


@pytest.mark.parametrize(
    "ext, promote_to_multi, expected_geometry_types, expected_geometry_type",
    [
        (".fgb", None, ["MultiPolygon"], "MultiPolygon"),
        (".fgb", True, ["MultiPolygon"], "MultiPolygon"),
        (".fgb", False, ["MultiPolygon", "Polygon"], "Unknown"),
        (".geojson", None, ["MultiPolygon", "Polygon"], "Unknown"),
        (".geojson", True, ["MultiPolygon"], "MultiPolygon"),
        (".geojson", False, ["MultiPolygon", "Polygon"], "Unknown"),
    ],
)
def test_write_dataframe_promote_to_multi(
    tmp_path,
    naturalearth_lowres,
    ext,
    promote_to_multi,
    expected_geometry_types,
    expected_geometry_type,
):
    input_gdf = read_dataframe(naturalearth_lowres)

    output_path = tmp_path / f"test_promote{ext}"
    write_dataframe(input_gdf, output_path, promote_to_multi=promote_to_multi)

    assert output_path.exists()
    output_gdf = read_dataframe(output_path)
    geometry_types = sorted(output_gdf.geometry.type.unique())
    assert geometry_types == expected_geometry_types
    assert read_info(output_path)["geometry_type"] == expected_geometry_type


@pytest.mark.parametrize(
    "ext, promote_to_multi, geometry_type, "
    "expected_geometry_types, expected_geometry_type",
    [
        (".fgb", None, "Unknown", ["MultiPolygon"], "Unknown"),
        (".geojson", False, "Unknown", ["MultiPolygon", "Polygon"], "Unknown"),
        (".geojson", None, "Unknown", ["MultiPolygon", "Polygon"], "Unknown"),
        (".geojson", None, "Polygon", ["MultiPolygon", "Polygon"], "Unknown"),
        (".geojson", None, "MultiPolygon", ["MultiPolygon", "Polygon"], "Unknown"),
        (".geojson", None, "Point", ["MultiPolygon", "Polygon"], "Unknown"),
        (".geojson", True, "Unknown", ["MultiPolygon"], "MultiPolygon"),
        (".gpkg", False, "Unknown", ["MultiPolygon", "Polygon"], "Unknown"),
        (".gpkg", None, "Unknown", ["MultiPolygon"], "Unknown"),
        (".gpkg", None, "Polygon", ["MultiPolygon"], "Polygon"),
        (".gpkg", None, "MultiPolygon", ["MultiPolygon"], "MultiPolygon"),
        (".gpkg", None, "Point", ["MultiPolygon"], "Point"),
        (".gpkg", True, "Unknown", ["MultiPolygon"], "Unknown"),
        (".shp", False, "Unknown", ["MultiPolygon", "Polygon"], "Polygon"),
        (".shp", None, "Unknown", ["MultiPolygon", "Polygon"], "Polygon"),
        (".shp", None, "Polygon", ["MultiPolygon", "Polygon"], "Polygon"),
        (".shp", None, "MultiPolygon", ["MultiPolygon", "Polygon"], "Polygon"),
        (".shp", True, "Unknown", ["MultiPolygon", "Polygon"], "Polygon"),
    ],
)
def test_write_dataframe_promote_to_multi_layer_geom_type(
    tmp_path,
    naturalearth_lowres,
    ext,
    promote_to_multi,
    geometry_type,
    expected_geometry_types,
    expected_geometry_type,
):
    input_gdf = read_dataframe(naturalearth_lowres)

    output_path = tmp_path / f"test_promote_layer_geom_type{ext}"

    if ext == ".gpkg" and geometry_type in ("Polygon", "Point"):
        ctx = pytest.warns(
            RuntimeWarning, match="A geometry of type MULTIPOLYGON is inserted"
        )
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        write_dataframe(
            input_gdf,
            output_path,
            promote_to_multi=promote_to_multi,
            geometry_type=geometry_type,
        )

    assert output_path.exists()
    output_gdf = read_dataframe(output_path)
    geometry_types = sorted(output_gdf.geometry.type.unique())
    assert geometry_types == expected_geometry_types
    assert read_info(output_path)["geometry_type"] == expected_geometry_type


@pytest.mark.parametrize(
    "ext, promote_to_multi, geometry_type, expected_raises_match",
    [
        (".fgb", False, "MultiPolygon", "Mismatched geometry type"),
        (".fgb", False, "Polygon", "Mismatched geometry type"),
        (".fgb", None, "Point", "Mismatched geometry type"),
        (".fgb", None, "Polygon", "Mismatched geometry type"),
        (".shp", None, "Point", "Could not add feature to layer at index"),
    ],
)
def test_write_dataframe_promote_to_multi_layer_geom_type_invalid(
    tmp_path,
    naturalearth_lowres,
    ext,
    promote_to_multi,
    geometry_type,
    expected_raises_match,
):
    input_gdf = read_dataframe(naturalearth_lowres)

    output_path = tmp_path / f"test{ext}"
    with pytest.raises(FeatureError, match=expected_raises_match):
        write_dataframe(
            input_gdf,
            output_path,
            promote_to_multi=promote_to_multi,
            geometry_type=geometry_type,
        )


def test_write_dataframe_layer_geom_type_invalid(tmp_path, naturalearth_lowres):
    df = read_dataframe(naturalearth_lowres)

    filename = tmp_path / "test.geojson"
    with pytest.raises(
        GeometryError, match="Geometry type is not supported: NotSupported"
    ):
        write_dataframe(df, filename, geometry_type="NotSupported")


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext not in ".shp"])
def test_write_dataframe_truly_mixed(tmp_path, ext):
    geometry = [
        shapely.Point(0, 0),
        shapely.LineString([(0, 0), (1, 1)]),
        shapely.box(0, 0, 1, 1),
        shapely.MultiPoint([shapely.Point(1, 1), shapely.Point(2, 2)]),
        shapely.MultiLineString(
            [shapely.LineString([(1, 1), (2, 2)]), shapely.LineString([(2, 2), (3, 3)])]
        ),
        shapely.MultiPolygon([shapely.box(1, 1, 2, 2), shapely.box(2, 2, 3, 3)]),
    ]

    df = gp.GeoDataFrame(
        {"col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, geometry=geometry, crs="EPSG:4326"
    )

    filename = tmp_path / f"test{ext}"

    if ext == ".fgb":
        # For .fgb, spatial_index=False to avoid the rows being reordered
        write_dataframe(df, filename, spatial_index=False)
    else:
        write_dataframe(df, filename)

    # Drivers that support mixed geometries will default to "Unknown" geometry type
    assert read_info(filename)["geometry_type"] == "Unknown"
    result = read_dataframe(filename)
    assert_geodataframe_equal(result, df, check_geom_type=True)


def test_write_dataframe_truly_mixed_invalid(tmp_path):
    # Shapefile doesn't support generic "Geometry" / "Unknown" type
    # for mixed geometries

    df = gp.GeoDataFrame(
        {"col": [1.0, 2.0, 3.0]},
        geometry=[
            shapely.Point(0, 0),
            shapely.LineString([(0, 0), (1, 1)]),
            shapely.box(0, 0, 1, 1),
        ],
        crs="EPSG:4326",
    )

    # ensure error message from GDAL is included
    msg = (
        "Could not add feature to layer at index 1: Attempt to "
        r"write non-point \(LINESTRING\) geometry to point shapefile."
    )
    with pytest.raises(FeatureError, match=msg):
        write_dataframe(df, tmp_path / "test.shp")


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext not in ".fgb"])
@pytest.mark.parametrize(
    "geoms",
    [
        [None, shapely.Point(1, 1)],
        [shapely.Point(1, 1), None],
        [None, shapely.Point(1, 1, 2)],
        [None, None],
    ],
)
def test_write_dataframe_infer_geometry_with_nulls(tmp_path, geoms, ext):
    filename = tmp_path / f"test{ext}"

    df = gp.GeoDataFrame({"col": [1.0, 2.0]}, geometry=geoms, crs="EPSG:4326")
    write_dataframe(df, filename)
    result = read_dataframe(filename)
    assert_geodataframe_equal(result, df)


@pytest.mark.filterwarnings(
    "ignore: You will likely lose important projection information"
)
def test_custom_crs_io(tmpdir, naturalearth_lowres_all_ext):
    df = read_dataframe(naturalearth_lowres_all_ext)
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


def test_write_read_mixed_column_values(tmp_path):
    mixed_values = ["test", 1.0, 1, datetime.now(), None, np.nan]
    geoms = [shapely.Point(0, 0) for _ in mixed_values]
    test_gdf = gp.GeoDataFrame(
        {"geometry": geoms, "mixed": mixed_values}, crs="epsg:31370"
    )
    output_path = tmp_path / "test_write_mixed_column.gpkg"
    write_dataframe(test_gdf, output_path)
    output_gdf = read_dataframe(output_path)
    assert len(test_gdf) == len(output_gdf)
    for idx, value in enumerate(mixed_values):
        if value in (None, np.nan):
            assert output_gdf["mixed"][idx] is None
        else:
            assert output_gdf["mixed"][idx] == str(value)


def test_write_read_null(tmp_path):
    output_path = tmp_path / "test_write_nan.gpkg"
    geom = shapely.Point(0, 0)
    test_data = {
        "geometry": [geom, geom, geom],
        "float64": [1.0, None, np.nan],
        "object_str": ["test", None, np.nan],
    }
    test_gdf = gp.GeoDataFrame(test_data, crs="epsg:31370")
    write_dataframe(test_gdf, output_path)
    result_gdf = read_dataframe(output_path)
    assert len(test_gdf) == len(result_gdf)
    assert result_gdf["float64"][0] == 1.0
    assert pd.isna(result_gdf["float64"][1])
    assert pd.isna(result_gdf["float64"][2])
    assert result_gdf["object_str"][0] == "test"
    assert result_gdf["object_str"][1] is None
    assert result_gdf["object_str"][2] is None


@pytest.mark.parametrize(
    "wkt,geom_types",
    [
        ("Point Z (0 0 0)", ["2.5D Point", "Point Z"]),
        ("LineString Z (0 0 0, 1 1 0)", ["2.5D LineString", "LineString Z"]),
        ("Polygon Z ((0 0 0, 0 1 0, 1 1 0, 0 0 0))", ["2.5D Polygon", "Polygon Z"]),
        ("MultiPoint Z (0 0 0, 1 1 0)", ["2.5D MultiPoint", "MultiPoint Z"]),
        (
            "MultiLineString Z ((0 0 0, 1 1 0), (2 2 2, 3 3 2))",
            ["2.5D MultiLineString", "MultiLineString Z"],
        ),
        (
            "MultiPolygon Z (((0 0 0, 0 1 0, 1 1 0, 0 0 0)), ((1 1 1, 1 2 1, 2 2 1, 1 1 1)))",  # NOQA
            ["2.5D MultiPolygon", "MultiPolygon Z"],
        ),
        (
            "GeometryCollection Z (Point Z (0 0 0))",
            ["2.5D GeometryCollection", "GeometryCollection Z"],
        ),
    ],
)
def test_write_geometry_z_types(tmp_path, wkt, geom_types):
    filename = tmp_path / "test.fgb"
    gdf = gp.GeoDataFrame(geometry=from_wkt([wkt]), crs="EPSG:4326")
    for geom_type in geom_types:
        write_dataframe(gdf, filename, geometry_type=geom_type)
        df = read_dataframe(filename)
        assert_geodataframe_equal(df, gdf)


@pytest.mark.parametrize("ext", ALL_EXTS)
@pytest.mark.parametrize(
    "test_descr, exp_geometry_type, mixed_dimensions, wkt",
    [
        ("1 Point Z", "Point Z", False, ["Point Z (0 0 0)"]),
        ("1 LineString Z", "LineString Z", False, ["LineString Z (0 0 0, 1 1 0)"]),
        (
            "1 Polygon Z",
            "Polygon Z",
            False,
            ["Polygon Z ((0 0 0, 0 1 0, 1 1 0, 0 0 0))"],
        ),
        ("1 MultiPoint Z", "MultiPoint Z", False, ["MultiPoint Z (0 0 0, 1 1 0)"]),
        (
            "1 MultiLineString Z",
            "MultiLineString Z",
            False,
            ["MultiLineString Z ((0 0 0, 1 1 0), (2 2 2, 3 3 2))"],
        ),
        (
            "1 MultiLinePolygon Z",
            "MultiPolygon Z",
            False,
            [
                "MultiPolygon Z (((0 0 0, 0 1 0, 1 1 0, 0 0 0)), ((1 1 1, 1 2 1, 2 2 1, 1 1 1)))"  # noqa: E501
            ],
        ),
        (
            "1 GeometryCollection Z",
            "GeometryCollection Z",
            False,
            ["GeometryCollection Z (Point Z (0 0 0))"],
        ),
        ("Point Z + Point", "Point Z", True, ["Point Z (0 0 0)", "Point (0 0)"]),
        ("Point Z + None", "Point Z", False, ["Point Z (0 0 0)", None]),
        (
            "Point Z + LineString Z",
            "Unknown",
            False,
            ["LineString Z (0 0 0, 1 1 0)", "Point Z (0 0 0)"],
        ),
        (
            "Point Z + LineString",
            "Unknown",
            True,
            ["LineString (0 0, 1 1)", "Point Z (0 0 0)"],
        ),
    ],
)
def test_write_geometry_z_types_auto(
    tmp_path, ext, test_descr, exp_geometry_type, mixed_dimensions, wkt
):
    # Shapefile has some different behaviour that other file types
    if ext == ".shp":
        if exp_geometry_type in ("GeometryCollection Z", "Unknown"):
            pytest.skip(f"ext {ext} doesn't support {exp_geometry_type}")
        elif exp_geometry_type == "MultiLineString Z":
            exp_geometry_type = "LineString Z"
        elif exp_geometry_type == "MultiPolygon Z":
            exp_geometry_type = "Polygon Z"

    column_data = {}
    column_data["test_descr"] = [test_descr] * len(wkt)
    column_data["idx"] = [str(idx) for idx in range(len(wkt))]
    gdf = gp.GeoDataFrame(column_data, geometry=from_wkt(wkt), crs="EPSG:4326")
    filename = tmp_path / f"test{ext}"

    if ext == ".fgb":
        # writing empty / null geometries not allowed by FlatGeobuf for
        # GDAL >= 3.6.4 and were simply not written previously
        gdf = gdf.loc[~(gdf.geometry.isna() | gdf.geometry.is_empty)]

    if mixed_dimensions and DRIVERS[ext] in DRIVERS_NO_MIXED_DIMENSIONS:
        with pytest.raises(
            DataSourceError,
            match=("Mixed 2D and 3D coordinates are not supported by"),
        ):
            write_dataframe(gdf, filename)
        return
    else:
        write_dataframe(gdf, filename)

    info = read_info(filename)
    assert info["geometry_type"] == exp_geometry_type

    result_gdf = read_dataframe(filename)
    if ext == ".geojsonl":
        result_gdf.crs = "EPSG:4326"

    assert_geodataframe_equal(gdf, result_gdf)


def test_read_multisurface(data_dir):
    df = read_dataframe(data_dir / "test_multisurface.gpkg")

    # MultiSurface should be converted to MultiPolygon
    assert df.geometry.type.tolist() == ["MultiPolygon"]


def test_read_dataset_kwargs(data_dir, use_arrow):
    filename = data_dir / "test_nested.geojson"

    # by default, nested data are not flattened
    df = read_dataframe(filename, use_arrow=use_arrow)

    expected = gp.GeoDataFrame(
        {
            "top_level": ["A"],
            "intermediate_level": ['{ "bottom_level": "B" }'],
        },
        geometry=[shapely.Point(0, 0)],
        crs="EPSG:4326",
    )

    assert_geodataframe_equal(df, expected)

    df = read_dataframe(filename, use_arrow=use_arrow, FLATTEN_NESTED_ATTRIBUTES="YES")

    expected = gp.GeoDataFrame(
        {
            "top_level": ["A"],
            "intermediate_level_bottom_level": ["B"],
        },
        geometry=[shapely.Point(0, 0)],
        crs="EPSG:4326",
    )

    assert_geodataframe_equal(df, expected)


def test_read_invalid_dataset_kwargs(naturalearth_lowres, use_arrow):
    with pytest.warns(RuntimeWarning, match="does not support open option INVALID"):
        read_dataframe(naturalearth_lowres, use_arrow=use_arrow, INVALID="YES")


def test_write_nullable_dtypes(tmp_path):
    path = tmp_path / "test_nullable_dtypes.gpkg"
    test_data = {
        "col1": pd.Series([1, 2, 3], dtype="int64"),
        "col2": pd.Series([1, 2, None], dtype="Int64"),
        "col3": pd.Series([0.1, None, 0.3], dtype="Float32"),
        "col4": pd.Series([True, False, None], dtype="boolean"),
        "col5": pd.Series(["a", None, "b"], dtype="string"),
    }
    input_gdf = gp.GeoDataFrame(
        test_data, geometry=[shapely.Point(0, 0)] * 3, crs="epsg:31370"
    )
    write_dataframe(input_gdf, path)
    output_gdf = read_dataframe(path)
    # We read it back as default (non-nullable) numpy dtypes, so we cast
    # to those for the expected result
    expected = input_gdf.copy()
    expected["col2"] = expected["col2"].astype("float64")
    expected["col3"] = expected["col3"].astype("float32")
    expected["col4"] = expected["col4"].astype("float64")
    expected["col5"] = expected["col5"].astype(object)
    assert_geodataframe_equal(output_gdf, expected)


@pytest.mark.parametrize(
    "metadata_type", ["dataset_metadata", "layer_metadata", "metadata"]
)
def test_metadata_io(tmpdir, naturalearth_lowres, metadata_type):
    metadata = {"level": metadata_type}

    df = read_dataframe(naturalearth_lowres)

    filename = os.path.join(str(tmpdir), "test.gpkg")
    write_dataframe(df, filename, **{metadata_type: metadata})

    metadata_key = "layer_metadata" if metadata_type == "metadata" else metadata_type

    assert read_info(filename)[metadata_key] == metadata


@pytest.mark.parametrize("metadata_type", ["dataset_metadata", "layer_metadata"])
@pytest.mark.parametrize(
    "metadata",
    [
        {1: 2},
        {"key": None},
        {"key": 1},
    ],
)
def test_invalid_metadata(tmpdir, naturalearth_lowres, metadata_type, metadata):
    with pytest.raises(ValueError, match="must be a string"):
        filename = os.path.join(str(tmpdir), "test.gpkg")
        write_dataframe(
            read_dataframe(naturalearth_lowres), filename, **{metadata_type: metadata}
        )


@pytest.mark.parametrize("metadata_type", ["dataset_metadata", "layer_metadata"])
def test_metadata_unsupported(tmpdir, naturalearth_lowres, metadata_type):
    """metadata is silently ignored"""

    filename = os.path.join(str(tmpdir), "test.geojson")
    write_dataframe(
        read_dataframe(naturalearth_lowres),
        filename,
        **{metadata_type: {"key": "value"}},
    )

    metadata_key = "layer_metadata" if metadata_type == "metadata" else metadata_type

    assert read_info(filename)[metadata_key] is None


@pytest.mark.skipif(not PANDAS_GE_15, reason="ArrowDtype requires pandas 1.5+")
def test_read_dataframe_arrow_dtypes(tmp_path):
    # https://github.com/geopandas/pyogrio/issues/319 - ensure arrow binary
    # column can be converted with from_wkb in case of missing values
    pytest.importorskip("pyarrow")
    filename = tmp_path / "test.gpkg"
    df = gp.GeoDataFrame(
        {"col": [1.0, 2.0]}, geometry=[Point(1, 1), None], crs="EPSG:4326"
    )
    write_dataframe(df, filename)

    result = read_dataframe(
        filename,
        use_arrow=True,
        arrow_to_pandas_kwargs={
            "types_mapper": lambda pa_dtype: pd.ArrowDtype(pa_dtype)
        },
    )
    assert isinstance(result["col"].dtype, pd.ArrowDtype)
    result["col"] = result["col"].astype("float64")
    assert_geodataframe_equal(result, df)
