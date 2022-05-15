import os

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import pytest

from pyogrio import list_layers, read_info, __gdal_geos_version__
from pyogrio.errors import DataLayerError, DataSourceError, FeatureError, GeometryError
from pyogrio.geopandas import read_dataframe, write_dataframe
from pyogrio.raw import (
    DRIVERS,
    DRIVERS_NO_MIXED_SINGLE_MULTI,
)
from pyogrio.tests.conftest import ALL_EXTS

try:
    import geopandas as gp
    from geopandas.testing import assert_geodataframe_equal

    has_geopandas = True
except ImportError:
    has_geopandas = False


# Note: this will also be false for GDAL < 3.4 when GEOS may be present but we
# cannot verify it
has_geos = __gdal_geos_version__ is not None


pytestmark = pytest.mark.skipif(not has_geopandas, reason="GeoPandas not available")


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

    assert df.geometry.iloc[0].type in ["Polygon", "MultiPolygon"]


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
    assert df.SEGMENT_NAME.isnull().max() == True
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
    if naturalearth_lowres_all_ext.suffix in [".gpkg"]:
        # Geopackage doesn't raise, but returns empty df?
        gdf = read_dataframe(naturalearth_lowres_all_ext, where="invalid")
        assert len(gdf) == 0
    else:
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

    df = read_dataframe(naturalearth_lowres_all_ext, bbox=(-140, 20, -100, 40))
    assert len(df) == 2
    assert np.array_equal(df.iso_a3, ["USA", "MEX"])


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
    assert sorted(df.iso_a3.tolist()) == ["CAN", "MEX", "USA"]

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
    assert sorted(df.iso_a3_renamed.tolist()) == ["CAN", "MEX", "USA"]


def test_read_sql_columns_where_bbox(naturalearth_lowres_all_ext):
    sql = "SELECT iso_a3 AS iso_a3_renamed, name, pop_est FROM naturalearth_lowres"
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        sql=sql,
        sql_dialect="OGRSQL",
        columns=["iso_a3_renamed", "name"],
        where="iso_a3_renamed IN ('CAN', 'USA', 'MEX')",
        bbox=(-140, 20, -100, 40),
    )
    assert len(df.columns) == 3
    assert len(df) == 2
    assert sorted(df.iso_a3_renamed.tolist()) == ["MEX", "USA"]


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


@pytest.mark.parametrize(
    "ext",
    ALL_EXTS,
)
def test_write_dataframe(tmp_path, naturalearth_lowres, ext):
    input_gdf = read_dataframe(naturalearth_lowres)
    output_path = tmp_path / f"test{ext}"

    if ext == ".fgb":
        # For .fgb, spatial_index=False to evade the rows being reordered
        write_dataframe(input_gdf, output_path, spatial_index=False)
    else:
        write_dataframe(input_gdf, output_path)

    assert output_path.exists()
    result_gdf = read_dataframe(output_path)

    geometry_types = result_gdf.geometry.type.unique()
    if DRIVERS[ext] in DRIVERS_NO_MIXED_SINGLE_MULTI:
        assert len(geometry_types) == 1
    else:
        assert len(geometry_types) == 2

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


def to_multipolygon(geometries):
    """
    Convert single part polygons to multipolygons.
    Parameters
    ----------
    geometries : ndarray of pygeos geometries
        can be mixed polygon and multipolygon types
    Returns
    -------
    ndarray of pygeos geometries, all multipolygon types
    """
    import pygeos

    ix = pygeos.get_type_id(geometries) == 3
    if ix.sum():
        geometries = geometries.copy()
        geometries[ix] = np.apply_along_axis(
            pygeos.multipolygons, arr=(np.expand_dims(geometries[ix], 1)), axis=1
        )
    return geometries


@pytest.mark.filterwarnings("ignore:.*Layer .* does not have any features to read")
@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext not in ".geojsonl"])
def test_write_empty_dataframe(tmp_path, ext):
    expected = gp.GeoDataFrame(geometry=[], crs=4326)

    filename = tmp_path / f"test{ext}"
    write_dataframe(expected, filename)

    assert filename.exists()
    df = read_dataframe(filename)
    assert_geodataframe_equal(df, expected)


def test_write_empty_dataframe_unsupported(tmp_path):
    # Writing empty dataframe to .geojsons results in a 0 byte file, which
    # is invalid to read again.
    expected = gp.GeoDataFrame(geometry=[], crs=4326)

    filename = tmp_path / "test.geojsonl"
    write_dataframe(expected, filename)

    assert filename.exists()
    with pytest.raises(
        Exception, match=".* not recognized as a supported file format."
    ):
        _ = read_dataframe(filename)


def test_write_dataframe_gdalparams(tmp_path, naturalearth_lowres):
    original_df = read_dataframe(naturalearth_lowres)

    test_noindex_filename = tmp_path / "test_gdalparams_noindex.shp"
    write_dataframe(original_df, test_noindex_filename, SPATIAL_INDEX="NO")
    assert test_noindex_filename.exists() is True
    test_noindex_index_filename = tmp_path / "test_gdalparams_noindex.qix"
    assert test_noindex_index_filename.exists() is False

    test_withindex_filename = tmp_path / "test_gdalparams_withindex.shp"
    write_dataframe(original_df, test_withindex_filename, SPATIAL_INDEX="YES")
    assert test_withindex_filename.exists() is True
    test_withindex_index_filename = tmp_path / "test_gdalparams_withindex.qix"
    assert test_withindex_index_filename.exists() is True


def test_write_dataframe_promote_to_multi_geojson(tmp_path, naturalearth_lowres):
    """Test .geojson, wich supports mixed multi and single geometries."""
    input_gdf = read_dataframe(naturalearth_lowres)

    # promote_to_multi=None (=default): no promotion for .geojson
    output_path = tmp_path / "test_promote_None.geojson"
    write_dataframe(input_gdf, output_path)

    assert output_path.exists()
    output_gdf = read_dataframe(output_path)
    geometry_types = output_gdf.geometry.type.unique()
    assert len(geometry_types) == 2

    # promote_to_multi=True: force promotion
    output_path = tmp_path / "test_promote.geojson"
    write_dataframe(input_gdf, output_path, promote_to_multi=True)

    assert output_path.exists()
    output_gdf = read_dataframe(output_path)
    geometry_types = output_gdf.geometry.type.unique()
    assert len(geometry_types) == 1


def test_write_dataframe_promote_to_multi_fgb(tmp_path, naturalearth_lowres):
    """Test .fgb, which needs promotion to save mixed multi and single geometries."""
    input_gdf = read_dataframe(naturalearth_lowres)

    # promote_to_multi=None (=default), promotion for .fgb
    output_path = tmp_path / "test_promote_None.fgb"
    write_dataframe(input_gdf, output_path)

    assert output_path.exists()
    output_gdf = read_dataframe(output_path)
    geometry_types = output_gdf.geometry.type.unique()
    assert len(geometry_types) == 1
    output_info = read_info(output_path)
    assert output_info["geometry_type"] == "MultiPolygon"

    # promote_to_multi=True: force promotion
    output_path = tmp_path / "test_promote_True.fgb"
    input_single_gdf = input_gdf[input_gdf.geom_type == "Polygon"]
    write_dataframe(input_single_gdf, output_path, promote_to_multi=True)

    assert output_path.exists()
    output_gdf = read_dataframe(output_path)
    geometry_types = output_gdf.geometry.type.unique()
    assert len(geometry_types) == 1
    output_info = read_info(output_path)
    assert output_info["geometry_type"] == "MultiPolygon"

    # promote_to_multi=False: prohibit promotion
    output_path = tmp_path / "test_promote_False.fgb"
    write_dataframe(input_gdf, output_path, promote_to_multi=False)

    assert output_path.exists()
    output_gdf = read_dataframe(output_path)
    geometry_types = output_gdf.geometry.type.unique()
    assert len(geometry_types) == 2
    output_info = read_info(output_path)
    assert output_info["geometry_type"] == "Unknown"


def test_write_dataframe_geometry_type_unknown(tmp_path, naturalearth_lowres):
    input_gdf = read_dataframe(naturalearth_lowres)

    # Without forced unknown
    output_path = tmp_path / "test_no_unknown.fgb"
    write_dataframe(input_gdf, output_path)

    assert output_path.exists()
    output_info = read_info(output_path)
    assert output_info["geometry_type"] == "MultiPolygon"
    output_gdf = read_dataframe(output_path)
    geometry_types = output_gdf.geometry.type.unique()
    assert len(geometry_types) == 1
    assert geometry_types[0] == "MultiPolygon"

    # With forced unknown
    output_path = tmp_path / "test_unknown.fgb"
    write_dataframe(input_gdf, output_path, layer_geometry_type="Unknown")

    assert output_path.exists()
    output_gdf = read_dataframe(output_path)
    output_info = read_info(output_path)
    assert output_info["geometry_type"] == "Unknown"
    # No promotion should be done
    geometry_types = output_gdf.geometry.type.unique()
    assert len(geometry_types) == 1


def test_write_dataframe_geometry_types(tmp_path, naturalearth_lowres):
    df = read_dataframe(naturalearth_lowres)

    filename = tmp_path / "test.gpkg"
    write_dataframe(df, filename, layer_geometry_type="Unknown")
    assert read_info(filename)["geometry_type"] == "Unknown"

    write_dataframe(df, filename, layer_geometry_type="Polygon")
    assert read_info(filename)["geometry_type"] == "Polygon"

    write_dataframe(df, filename, layer_geometry_type="MultiPolygon")
    assert read_info(filename)["geometry_type"] == "MultiPolygon"

    write_dataframe(df, filename, layer_geometry_type="LineString")
    assert read_info(filename)["geometry_type"] == "LineString"

    write_dataframe(df, filename, layer_geometry_type="Point")
    assert read_info(filename)["geometry_type"] == "Point"

    with pytest.raises(
        GeometryError, match="Geometry type is not supported: NotSupported"
    ):
        write_dataframe(df, filename, layer_geometry_type="NotSupported")


@pytest.mark.parametrize(
    "ext",
    [ext for ext in ALL_EXTS if ext not in ".shp"],
)
def test_write_dataframe_truly_mixed(tmp_path, ext):
    from shapely.geometry import Point, LineString, box

    df = gp.GeoDataFrame(
        {"col": [1.0, 2.0, 3.0]},
        geometry=[Point(0, 0), LineString([(0, 0), (1, 1)]), box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    filename = tmp_path / f"test{ext}"

    if ext == ".fgb":
        # For .fgb, spatial_index=False to evade the rows being reordereds
        write_dataframe(df, filename, spatial_index=False)
    else:
        write_dataframe(df, filename)

    # Drivers that support mixed geometries will default to "Unknown" geometry type
    assert read_info(filename)["geometry_type"] == "Unknown"
    result = read_dataframe(filename)
    assert_geodataframe_equal(result, df)


def test_write_dataframe_truly_mixed_unsupported(tmp_path):
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
