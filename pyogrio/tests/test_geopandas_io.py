from datetime import datetime
import os

import numpy as np
import pytest

from pyogrio import list_layers, read_info, __gdal_version__, __gdal_geos_version__
from pyogrio.errors import DataLayerError, DataSourceError, FeatureError, GeometryError
from pyogrio.geopandas import read_dataframe, write_dataframe
from pyogrio.raw import DRIVERS, DRIVERS_NO_MIXED_SINGLE_MULTI
from pyogrio.tests.conftest import ALL_EXTS

try:
    import pandas as pd
    from pandas.testing import assert_frame_equal, assert_index_equal

    import geopandas as gp
    from geopandas.testing import assert_geodataframe_equal

    from shapely.geometry import Point
except ImportError:
    pass


pytest.importorskip("geopandas")


# Note: this will also be false for GDAL < 3.4 when GEOS may be present but we
# cannot verify it
has_geos = __gdal_geos_version__ is not None


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
    "naturalearth_lowres, expected_ext",
    [(".gpkg", ".gpkg"), (".shp", ".shp")],
    indirect=["naturalearth_lowres"],
)
def test_fixture_naturalearth_lowres(naturalearth_lowres, expected_ext):
    # Test the fixture with "indirect" parameter
    assert naturalearth_lowres.suffix == expected_ext
    df = read_dataframe(naturalearth_lowres)
    assert len(df) == 177


def test_read_no_geometry(naturalearth_lowres_all_ext):
    df = read_dataframe(naturalearth_lowres_all_ext, read_geometry=False)
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


def test_read_layer_invalid(naturalearth_lowres_all_ext):
    with pytest.raises(DataLayerError, match="Layer 'wrong' could not be opened"):
        read_dataframe(naturalearth_lowres_all_ext, layer="wrong")


@pytest.mark.filterwarnings("ignore: Measured")
def test_read_datetime(test_fgdb_vsi):
    df = read_dataframe(test_fgdb_vsi, layer="test_lines", max_features=1)
    assert df.SURVEY_DAT.dtype.name == "datetime64[ns]"


def test_read_null_values(test_fgdb_vsi):
    df = read_dataframe(test_fgdb_vsi, read_geometry=False)

    # make sure that Null values are preserved
    assert df.SEGMENT_NAME.isnull().max()
    assert df.loc[df.SEGMENT_NAME.isnull()].SEGMENT_NAME.iloc[0] is None


def test_read_fid_as_index(naturalearth_lowres_all_ext):
    kwargs = {"skip_features": 2, "max_features": 2}

    # default is to not set FIDs as index
    df = read_dataframe(naturalearth_lowres_all_ext, **kwargs)
    assert_index_equal(df.index, pd.RangeIndex(0, 2))

    df = read_dataframe(naturalearth_lowres_all_ext, fid_as_index=False, **kwargs)
    assert_index_equal(df.index, pd.RangeIndex(0, 2))

    df = read_dataframe(naturalearth_lowres_all_ext, fid_as_index=True, **kwargs)
    if naturalearth_lowres_all_ext.suffix in [".gpkg"]:
        # File format where fid starts at 1
        assert_index_equal(df.index, pd.Index([3, 4], name="fid"))
    else:
        # File format where fid starts at 0
        assert_index_equal(df.index, pd.Index([2, 3], name="fid"))


@pytest.mark.filterwarnings("ignore:.*Layer .* does not have any features to read")
def test_read_where(naturalearth_lowres_all_ext):
    # empty filter should return full set of records
    df = read_dataframe(naturalearth_lowres_all_ext, where="")
    assert len(df) == 177

    # should return singular item
    df = read_dataframe(naturalearth_lowres_all_ext, where="iso_a3 = 'CAN'")
    assert len(df) == 1
    assert df.iloc[0].iso_a3 == "CAN"

    df = read_dataframe(
        naturalearth_lowres_all_ext, where="iso_a3 IN ('CAN', 'USA', 'MEX')"
    )
    assert len(df) == 3
    assert len(set(df.iso_a3.unique()).difference(["CAN", "USA", "MEX"])) == 0

    # should return items within range
    df = read_dataframe(
        naturalearth_lowres_all_ext, where="POP_EST >= 10000000 AND POP_EST < 100000000"
    )
    assert len(df) == 75
    assert df.pop_est.min() >= 10000000
    assert df.pop_est.max() < 100000000

    # should match no items
    df = read_dataframe(naturalearth_lowres_all_ext, where="ISO_A3 = 'INVALID'")
    assert len(df) == 0


@pytest.mark.filterwarnings("ignore:.*Layer .* does not have any features to read")
def test_read_where_invalid(naturalearth_lowres_all_ext):
    with pytest.raises(ValueError, match="Invalid SQL"):
        read_dataframe(naturalearth_lowres_all_ext, where="invalid")


@pytest.mark.parametrize("bbox", [(1,), (1, 2), (1, 2, 3)])
def test_read_bbox_invalid(naturalearth_lowres_all_ext, bbox):
    with pytest.raises(ValueError, match="Invalid bbox"):
        read_dataframe(naturalearth_lowres_all_ext, bbox=bbox)


def test_read_bbox(naturalearth_lowres_all_ext):
    # should return no features
    with pytest.warns(UserWarning, match="does not have any features to read"):
        df = read_dataframe(naturalearth_lowres_all_ext, bbox=(0, 0, 0.00001, 0.00001))
        assert len(df) == 0

    df = read_dataframe(naturalearth_lowres_all_ext, bbox=(-85, 8, -80, 10))
    assert len(df) == 2

    assert np.array_equal(df.iso_a3, ["PAN", "CRI"])


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


def test_read_non_existent_file():
    # ensure consistent error type / message from GDAL
    with pytest.raises(DataSourceError, match="No such file or directory"):
        read_dataframe("non-existent.shp")

    with pytest.raises(DataSourceError, match="does not exist in the file system"):
        read_dataframe("/vsizip/non-existent.zip")

    with pytest.raises(DataSourceError, match="does not exist in the file system"):
        read_dataframe("zip:///non-existent.zip")


@pytest.mark.filterwarnings("ignore:.*Layer .* does not have any features to read")
def test_read_sql(naturalearth_lowres_all_ext):
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
    with pytest.raises(ValueError, match="'skip_features' must be between 0 and 0"):
        _ = read_dataframe(
            naturalearth_lowres_all_ext, sql=sql, skip_features=1, sql_dialect="OGRSQL"
        )


@pytest.mark.skipif(not has_geos, reason="Spatial SQL operations require GEOS")
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


@pytest.mark.skipif(not has_geos, reason="Spatial SQL operations require GEOS")
@pytest.mark.parametrize(
    "naturalearth_lowres", [".gpkg"], indirect=["naturalearth_lowres"]
)
def test_read_sql_dialect_sqlite_gpkg(naturalearth_lowres):
    # "INDIRECT_SQL" prohibits GDAL from passing the sql statement to sqlite.
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
    is_json = ext in [".json", ".geojson", ".geojsonl"]
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


@pytest.mark.filterwarnings("ignore:.*Layer .* does not have any features to read")
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
    from shapely.geometry import (
        box,
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
    )

    geometry = [
        Point(0, 0),
        LineString([(0, 0), (1, 1)]),
        box(0, 0, 1, 1),
        MultiPoint([Point(1, 1), Point(2, 2)]),
        MultiLineString([LineString([(1, 1), (2, 2)]), LineString([(2, 2), (3, 3)])]),
        MultiPolygon([box(1, 1, 2, 2), box(2, 2, 3, 3)]),
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
    from shapely.geometry import Point, LineString, box

    df = gp.GeoDataFrame(
        {"col": [1.0, 2.0, 3.0]},
        geometry=[Point(0, 0), LineString([(0, 0), (1, 1)]), box(0, 0, 1, 1)],
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
    [[None, Point(1, 1)], [Point(1, 1), None], [None, Point(1, 1, 2)], [None, None]],
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
    from shapely.geometry import Point

    mixed_values = ["test", 1.0, 1, datetime.now(), None, np.nan]
    geoms = [Point(0, 0) for _ in mixed_values]
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
    from shapely.geometry import Point

    output_path = tmp_path / "test_write_nan.gpkg"
    geom = Point(0, 0)
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
    from geopandas.array import from_wkt

    filename = tmp_path / "test.fgb"

    gdf = gp.GeoDataFrame(geometry=from_wkt([wkt]), crs="EPSG:4326")
    for geom_type in geom_types:
        write_dataframe(gdf, filename, geometry_type=geom_type)
        df = read_dataframe(filename)
        assert_geodataframe_equal(df, gdf)


def test_read_multisurface(data_dir):
    df = read_dataframe(data_dir / "test_multisurface.gpkg")

    # MultiSurface should be converted to MultiPolygon
    assert df.geometry.type.tolist() == ["MultiPolygon"]
