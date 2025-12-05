import contextlib
import locale
import os
import re
import warnings
from datetime import datetime
from io import BytesIO
from zipfile import ZipFile

import numpy as np

from pyogrio import (
    __gdal_version__,
    list_drivers,
    list_layers,
    read_info,
    set_gdal_config_options,
    vsi_listtree,
    vsi_unlink,
)
from pyogrio._compat import (
    GDAL_GE_37,
    GDAL_GE_311,
    HAS_ARROW_WRITE_API,
    HAS_PYPROJ,
    PANDAS_GE_15,
    PANDAS_GE_23,
    PANDAS_GE_30,
    SHAPELY_GE_21,
)
from pyogrio.errors import DataLayerError, DataSourceError, FeatureError, GeometryError
from pyogrio.geopandas import PANDAS_GE_20, read_dataframe, write_dataframe
from pyogrio.raw import (
    DRIVERS_NO_MIXED_DIMENSIONS,
    DRIVERS_NO_MIXED_SINGLE_MULTI,
)
from pyogrio.tests.conftest import (
    ALL_EXTS,
    DRIVERS,
    GDAL_HAS_PARQUET_DRIVER,
    START_FID,
    requires_arrow_write_api,
    requires_gdal_geos,
    requires_pyarrow_api,
    requires_pyproj,
)

import pytest

try:
    import geopandas as gp
    import pandas as pd
    from geopandas.array import from_wkt
    from pandas.api.types import (
        is_bool_dtype,
        is_datetime64_dtype,
        is_float_dtype,
        is_integer_dtype,
        is_object_dtype,
        is_string_dtype,
    )

    import shapely  # if geopandas is present, shapely is expected to be present
    from shapely.geometry import Point

    from geopandas.testing import assert_geodataframe_equal
    from pandas.testing import (
        assert_index_equal,
        assert_series_equal,
    )

except ImportError:
    pass


pytest.importorskip("geopandas")


@pytest.fixture(
    scope="session",
    params=[
        False,
        pytest.param(True, marks=requires_pyarrow_api),
    ],
)
def use_arrow(request):
    return request.param


@pytest.fixture(autouse=True)
def skip_if_no_arrow_write_api(request):
    # automatically skip tests with use_arrow=True and that require Arrow write
    # API (marked with `@pytest.mark.requires_arrow_write_api`) if it is not available
    use_arrow = (
        request.getfixturevalue("use_arrow")
        if "use_arrow" in request.fixturenames
        else False
    )
    if (
        use_arrow
        and not HAS_ARROW_WRITE_API
        and request.node.get_closest_marker("requires_arrow_write_api")
    ):
        pytest.skip("GDAL>=3.8 required for Arrow write API")


@contextlib.contextmanager
def use_arrow_context():
    original = os.environ.get("PYOGRIO_USE_ARROW", None)
    os.environ["PYOGRIO_USE_ARROW"] = "1"
    yield
    if original:
        os.environ["PYOGRIO_USE_ARROW"] = original
    else:
        del os.environ["PYOGRIO_USE_ARROW"]


def test_spatialite_available(test_gpkg_nulls):
    """Check if SpatiaLite is available by running a simple SQL query."""
    _ = read_dataframe(
        test_gpkg_nulls, sql="select spatialite_version();", sql_dialect="SQLITE"
    )


@pytest.mark.parametrize(
    "encoding, arrow",
    [
        ("utf-8", False),
        pytest.param("utf-8", True, marks=requires_pyarrow_api),
        ("cp1252", False),
        (None, False),
    ],
)
def test_read_csv_encoding(tmp_path, encoding, arrow):
    """ "Test reading CSV files with different encodings.

    Arrow only supports utf-8 encoding.
    """
    # Write csv test file. Depending on the os this will be written in a different
    # encoding: for linux and macos this is utf-8, for windows it is cp1252.
    csv_path = tmp_path / "test.csv"
    with open(csv_path, "w", encoding=encoding) as csv:
        csv.write("näme,city\n")
        csv.write("Wilhelm Röntgen,Zürich\n")

    # Read csv. The data should be read with the same default encoding as the csv file
    # was written in, but should have been converted to utf-8 in the dataframe returned.
    # Hence, the asserts below, with strings in utf-8, be OK.
    df = read_dataframe(csv_path, encoding=encoding, use_arrow=arrow)

    assert len(df) == 1
    assert df.columns.tolist() == ["näme", "city"]
    assert df.city.tolist() == ["Zürich"]
    assert df.näme.tolist() == ["Wilhelm Röntgen"]


@pytest.mark.skipif(
    locale.getpreferredencoding().upper() == "UTF-8",
    reason="test requires non-UTF-8 default platform",
)
def test_read_csv_platform_encoding(tmp_path, use_arrow):
    """Verify that read defaults to platform encoding; only works on Windows (CP1252).

    When use_arrow=True, reading an non-UTF8 fails.
    """
    csv_path = tmp_path / "test.csv"
    with open(csv_path, "w", encoding=locale.getpreferredencoding()) as csv:
        csv.write("näme,city\n")
        csv.write("Wilhelm Röntgen,Zürich\n")

    if use_arrow:
        with pytest.raises(
            DataSourceError,
            match="; please use_arrow=False",
        ):
            df = read_dataframe(csv_path, use_arrow=use_arrow)
    else:
        df = read_dataframe(csv_path, use_arrow=use_arrow)

        assert len(df) == 1
        assert df.columns.tolist() == ["näme", "city"]
        assert df.city.tolist() == ["Zürich"]
        assert df.näme.tolist() == ["Wilhelm Röntgen"]


def test_read_dataframe(naturalearth_lowres_all_ext):
    df = read_dataframe(naturalearth_lowres_all_ext)

    if HAS_PYPROJ:
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


def test_read_dataframe_vsi(naturalearth_lowres_vsi, use_arrow):
    df = read_dataframe(naturalearth_lowres_vsi[1], use_arrow=use_arrow)
    assert len(df) == 177


@pytest.mark.parametrize(
    "columns, fid_as_index, exp_len", [(None, False, 3), ([], True, 3), ([], False, 0)]
)
def test_read_layer_without_geometry(
    no_geometry_file, columns, fid_as_index, use_arrow, exp_len
):
    result = read_dataframe(
        no_geometry_file,
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


def test_read_force_2d(tmp_path, use_arrow):
    filename = tmp_path / "test.gpkg"

    # create a GPKG with 3D point values
    expected = gp.GeoDataFrame(
        geometry=[Point(0, 0, 0), Point(1, 1, 0)], crs="EPSG:4326"
    )
    write_dataframe(expected, filename)

    df = read_dataframe(filename)
    assert df.iloc[0].geometry.has_z

    df = read_dataframe(
        filename,
        force_2d=True,
        max_features=1,
        use_arrow=use_arrow,
    )
    assert not df.iloc[0].geometry.has_z


def test_read_geojson_error(naturalearth_lowres_geojson, use_arrow):
    try:
        set_gdal_config_options({"OGR_GEOJSON_MAX_OBJ_SIZE": 0.01})
        with pytest.raises(
            DataSourceError,
            match="Failed to read GeoJSON data; .* GeoJSON object too complex",
        ):
            read_dataframe(naturalearth_lowres_geojson, use_arrow=use_arrow)
    finally:
        set_gdal_config_options({"OGR_GEOJSON_MAX_OBJ_SIZE": None})


@pytest.mark.skipif(
    "LIBKML" not in list_drivers(),
    reason="LIBKML driver is not available and is needed to read simpledata element",
)
def test_read_kml_simpledata(kml_file, use_arrow):
    """Test reading a KML file with a simpledata element.

    Simpledata elements are only read by the LibKML driver, not the KML driver.
    """
    gdf = read_dataframe(kml_file, use_arrow=use_arrow)

    # Check if the simpledata column is present.
    assert "formation" in gdf.columns
    assert gdf["formation"].iloc[0] == "Ton"


def test_read_layer(tmp_path, use_arrow):
    filename = tmp_path / "test.gpkg"

    # create a multilayer GPKG
    expected1 = gp.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
    if use_arrow:
        # TODO this needs to be fixed on the geopandas side (to ensure the
        # GeoDataFrame() constructor does this), when use_arrow we already
        # get columns Index with string dtype
        expected1.columns = expected1.columns.astype("str")
    write_dataframe(
        expected1,
        filename,
        layer="layer1",
    )

    expected2 = gp.GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:4326")
    if use_arrow:
        expected2.columns = expected2.columns.astype("str")
    write_dataframe(expected2, filename, layer="layer2", append=True)

    assert np.array_equal(
        list_layers(filename), [["layer1", "Point"], ["layer2", "Point"]]
    )

    kwargs = {"use_arrow": use_arrow, "max_features": 1}

    # The first layer is read by default, which will warn when there are multiple
    # layers
    with pytest.warns(UserWarning, match="More than one layer found"):
        df = read_dataframe(filename, **kwargs)

    assert_geodataframe_equal(df, expected1)

    # Reading a specific layer by name should return that layer.
    # Detected here by a known column.
    df = read_dataframe(filename, layer="layer2", **kwargs)
    assert_geodataframe_equal(df, expected2)

    # Reading a specific layer by index should return that layer
    df = read_dataframe(filename, layer=1, **kwargs)
    assert_geodataframe_equal(df, expected2)


def test_read_layer_invalid(naturalearth_lowres_all_ext, use_arrow):
    with pytest.raises(DataLayerError, match="Layer 'wrong' could not be opened"):
        read_dataframe(naturalearth_lowres_all_ext, layer="wrong", use_arrow=use_arrow)


@pytest.mark.parametrize("columns", [None, [], ["col"]])
def test_read_datetime_columns(datetime_file, columns, use_arrow):
    df = read_dataframe(datetime_file, columns=columns, use_arrow=use_arrow)

    # Check result
    if columns is None or "col" in columns:
        assert "col" in df.columns
        assert is_datetime64_dtype(df.col.dtype)
        if PANDAS_GE_20:
            # starting with pandas 2.0, it preserves the passed datetime resolution
            assert df.col.dtype.name == "datetime64[ms]"
        else:
            assert df.col.dtype.name == "datetime64[ns]"
    else:
        assert len(df.columns) == 1  # only geometry


def test_read_list_types(list_field_values_files, use_arrow):
    """Test reading a geojson file containing fields with lists."""
    if list_field_values_files.suffix == ".parquet" and not GDAL_HAS_PARQUET_DRIVER:
        pytest.skip(
            "Skipping test for parquet as the GDAL Parquet driver is not available"
        )

    info = read_info(list_field_values_files)
    suffix = list_field_values_files.suffix

    result = read_dataframe(list_field_values_files, use_arrow=use_arrow)

    # Check list_int column
    assert "list_int" in result.columns
    assert info["fields"][1] == "list_int"
    assert info["ogr_types"][1] in ("OFTIntegerList", "OFTInteger64List")
    assert result["list_int"][0].tolist() == [0, 1]
    assert result["list_int"][1].tolist() == [2, 3]
    assert result["list_int"][2].tolist() == []
    assert result["list_int"][3] is None
    assert result["list_int"][4] is None

    # Check list_double column
    assert "list_double" in result.columns
    assert info["fields"][2] == "list_double"
    assert info["ogr_types"][2] == "OFTRealList"
    assert result["list_double"][0].tolist() == [0.0, 1.0]
    assert result["list_double"][1].tolist() == [2.0, 3.0]
    assert result["list_double"][2].tolist() == []
    assert result["list_double"][3] is None
    assert result["list_double"][4] is None

    # Check list_string column
    assert "list_string" in result.columns
    assert info["fields"][3] == "list_string"
    assert info["ogr_types"][3] == "OFTStringList"
    assert result["list_string"][0].tolist() == ["string1", "string2"]
    assert result["list_string"][1].tolist() == ["string3", "string4", ""]
    assert result["list_string"][2].tolist() == []
    assert result["list_string"][3] is None
    assert result["list_string"][4] == [""]

    # Check list_int_with_null column
    if suffix == ".geojson":
        # Once any row of a column contains a null value in a list, the column isn't
        # recognized as a list column anymore for .geojson files, but as a JSON column.
        # Because JSON columns containing JSON Arrays are also parsed to python lists,
        # the end result is the same...
        exp_type = "OFTString"
        exp_subtype = "OFSTJSON"
        exp_list_int_with_null_value = [0, None]
    else:
        # For .parquet files, the list column is preserved as a list column.
        exp_type = "OFTInteger64List"
        exp_subtype = "OFSTNone"
        if use_arrow:
            exp_list_int_with_null_value = [0.0, np.nan]
        else:
            exp_list_int_with_null_value = [0, 0]
            # xfail: when reading a list of int with None values without Arrow from a
            # .parquet file, the None values become 0, which is wrong.
            # https://github.com/OSGeo/gdal/issues/13448

    assert "list_int_with_null" in result.columns
    assert info["fields"][4] == "list_int_with_null"
    assert info["ogr_types"][4] == exp_type
    assert info["ogr_subtypes"][4] == exp_subtype
    assert result["list_int_with_null"][0][0] == 0
    if exp_list_int_with_null_value[1] == 0:
        assert result["list_int_with_null"][0][1] == exp_list_int_with_null_value[1]
    else:
        assert pd.isna(result["list_int_with_null"][0][1])

    if suffix == ".geojson":
        # For .geojson, the lists are already python lists
        assert result["list_int_with_null"][1] == [2, 3]
        assert result["list_int_with_null"][2] == []
    else:
        # For .parquet, the lists are numpy arrays
        assert result["list_int_with_null"][1].tolist() == [2, 3]
        assert result["list_int_with_null"][2].tolist() == []

    assert pd.isna(result["list_int_with_null"][3])
    assert pd.isna(result["list_int_with_null"][4])

    # Check list_string_with_null column
    if suffix == ".geojson":
        # Once any row of a column contains a null value in a list, the column isn't
        # recognized as a list column anymore for .geojson files, but as a JSON column.
        # Because JSON columns containing JSON Arrays are also parsed to python lists,
        # the end result is the same...
        exp_type = "OFTString"
        exp_subtype = "OFSTJSON"
    else:
        # For .parquet files, the list column is preserved as a list column.
        exp_type = "OFTStringList"
        exp_subtype = "OFSTNone"

    assert "list_string_with_null" in result.columns
    assert info["fields"][5] == "list_string_with_null"
    assert info["ogr_types"][5] == exp_type
    assert info["ogr_subtypes"][5] == exp_subtype

    if suffix == ".geojson":
        # For .geojson, the lists are already python lists
        assert result["list_string_with_null"][0] == ["string1", None]
        assert result["list_string_with_null"][1] == ["string3", "string4", ""]
        assert result["list_string_with_null"][2] == []
    else:
        # For .parquet, the lists are numpy arrays
        # When use_arrow=False, the None becomes an empty string, which is wrong.
        exp_value = ["string1", ""] if not use_arrow else ["string1", None]
        assert result["list_string_with_null"][0].tolist() == exp_value
        assert result["list_string_with_null"][1].tolist() == ["string3", "string4", ""]
        assert result["list_string_with_null"][2].tolist() == []

    assert pd.isna(result["list_string_with_null"][3])
    assert result["list_string_with_null"][4] == [""]


@pytest.mark.parametrize("columns", [None, [], ["list_int", "list_string"]])
def test_read_list_types_columns(request, list_field_values_files, use_arrow, columns):
    """Test reading a geojson file containing fields with lists."""
    if list_field_values_files.suffix == ".parquet" and not GDAL_HAS_PARQUET_DRIVER:
        pytest.skip(
            "Skipping test for parquet as the GDAL Parquet driver is not available"
        )
    if (
        use_arrow
        and columns
        and len(columns) == 2
        and list_field_values_files.suffix == ".parquet"
    ):
        # This gives following error, not sure why. Opened an issue for followup:
        # https://github.com/geopandas/pyogrio/issues/XXX
        error_msg = (
            "This fails with 'pyarrow.lib.ArrowInvalid: ArrowArray struct has "
            "1 children, expected 0 for type extension<geoarrow.wkb>'"
        )
        request.node.add_marker(pytest.mark.xfail(reason=error_msg))

    result = read_dataframe(
        list_field_values_files, use_arrow=use_arrow, columns=columns
    )

    # Check result
    exp_columns = 7 if columns is None else len(columns) + 1  # +1 for geometry
    assert len(result.columns) == exp_columns


@pytest.mark.requires_arrow_write_api
@pytest.mark.skipif(
    not GDAL_HAS_PARQUET_DRIVER, reason="Parquet driver is not available"
)
def test_read_list_nested_struct_parquet_file(
    list_nested_struct_parquet_file, use_arrow
):
    """Test reading a Parquet file containing nested struct and list types."""
    if not use_arrow:
        pytest.skip(
            "When use_arrow=False, gdal flattens nested columns to seperate columns. "
            "Not sure how we want to deal with this case, but for now just skip."
        )

    result = read_dataframe(list_nested_struct_parquet_file, use_arrow=use_arrow)

    assert "col_flat" in result.columns
    assert np.array_equal(result["col_flat"].to_numpy(), np.array([0, 1, 2]))

    assert "col_list" in result.columns
    assert result["col_list"].dtype == object
    assert result["col_list"][0].tolist() == [1, 2, 3]
    assert result["col_list"][1].tolist() == [1, 2, 3]
    assert result["col_list"][2].tolist() == [1, 2, 3]

    assert "col_nested" in result.columns
    assert result["col_nested"].dtype == object
    assert result["col_nested"][0].tolist() == [{"a": 1, "b": 2}, {"a": 1, "b": 2}]
    assert result["col_nested"][1].tolist() == [{"a": 1, "b": 2}, {"a": 1, "b": 2}]
    assert result["col_nested"][2].tolist() == [{"a": 1, "b": 2}, {"a": 1, "b": 2}]

    assert "col_struct" in result.columns
    assert result["col_struct"].dtype == object
    assert result["col_struct"][0] == {"a": 1, "b": 2}
    assert result["col_struct"][1] == {"a": 1, "b": 2}
    assert result["col_struct"][2] == {"a": 1, "b": 2}


@pytest.mark.requires_arrow_write_api
def test_roundtrip_many_data_types_geojson_file(
    request, tmp_path, many_data_types_geojson_file, use_arrow
):
    """Test roundtripping a GeoJSON file containing many data types."""

    def validate_result(df: pd.DataFrame, use_arrow: bool, ignore_mixed_list_col=False):
        """Function to validate the data of many_data_types_geojson_file.

        Depending on arrow being used or not there are small differences.
        """
        assert "int_col" in df.columns
        assert is_integer_dtype(df["int_col"].dtype)
        assert df["int_col"].to_list() == [1]

        assert "float_col" in df.columns
        assert is_float_dtype(df["float_col"].dtype)
        assert df["float_col"].to_list() == [1.5]

        assert "str_col" in df.columns
        assert is_string_dtype(df["str_col"].dtype)
        assert df["str_col"].to_list() == ["string"]

        assert "bool_col" in df.columns
        assert is_bool_dtype(df["bool_col"].dtype)
        assert df["bool_col"].to_list() == [True]

        assert "date_col" in df.columns
        if use_arrow:
            # Arrow returns dates as datetime.date objects.
            assert is_object_dtype(df["date_col"].dtype)
            assert df["date_col"].to_list() == [pd.Timestamp("2020-01-01").date()]
        else:
            # Without arrow, date columns are returned as datetime64.
            assert is_datetime64_dtype(df["date_col"].dtype)
            assert df["date_col"].to_list() == [pd.Timestamp("2020-01-01")]

        # Ignore time columns till this is solved:
        # Reported in https://github.com/geopandas/pyogrio/issues/615
        # assert "time_col" in df.columns
        # assert is_object_dtype(df["time_col"].dtype)
        # assert df["time_col"].to_list() == [time(12, 0, 0)]

        assert "datetime_col" in df.columns
        assert is_datetime64_dtype(df["datetime_col"].dtype)
        assert df["datetime_col"].to_list() == [pd.Timestamp("2020-01-01T12:00:00")]

        assert "list_int_col" in df.columns
        assert is_object_dtype(df["list_int_col"].dtype)
        assert df["list_int_col"][0].tolist() == [1, 2, 3]

        assert "list_str_col" in df.columns
        assert is_object_dtype(df["list_str_col"].dtype)
        assert df["list_str_col"][0].tolist() == ["a", "b", "c"]

        if not ignore_mixed_list_col:
            assert "list_mixed_col" in df.columns
            assert is_object_dtype(df["list_mixed_col"].dtype)
            assert df["list_mixed_col"][0] == [1, "a", None, True]

    # Read and validate result of reading
    read_gdf = read_dataframe(many_data_types_geojson_file, use_arrow=use_arrow)
    validate_result(read_gdf, use_arrow)

    # Write the data read, read it back, and validate again
    if use_arrow:
        # Writing a column with mixed types in a list is not supported with Arrow.
        ignore_mixed_list_col = True
        read_gdf = read_gdf.drop(columns=["list_mixed_col"])
    else:
        ignore_mixed_list_col = False
        request.node.add_marker(
            pytest.mark.xfail(
                reason="roundtripping list types fails with use_arrow=False"
            )
        )

    tmp_file = tmp_path / "temp.geojson"
    write_dataframe(read_gdf, tmp_file, use_arrow=use_arrow)

    # Validate data written
    read_back_gdf = read_dataframe(tmp_file, use_arrow=use_arrow)
    validate_result(
        read_back_gdf, use_arrow, ignore_mixed_list_col=ignore_mixed_list_col
    )


@pytest.mark.filterwarnings(
    "ignore: Non-conformant content for record 1 in column dates"
)
@pytest.mark.requires_arrow_write_api
def test_write_datetime_mixed_offset(tmp_path, use_arrow):
    # Australian Summer Time AEDT (GMT+11), Standard Time AEST (GMT+10)
    dates = ["2023-01-01 11:00:01.111", "2023-06-01 10:00:01.111"]
    naive_col = pd.Series(pd.to_datetime(dates), name="dates")
    localised_col = naive_col.dt.tz_localize("Australia/Sydney")
    utc_col = localised_col.dt.tz_convert("UTC")
    if PANDAS_GE_20:
        utc_col = utc_col.dt.as_unit("ms")


@pytest.mark.parametrize("datetime_as_string", [False, True])
@pytest.mark.parametrize("mixed_offsets_as_utc", [False, True])
def test_read_datetime_long_ago(
    geojson_datetime_long_ago, use_arrow, mixed_offsets_as_utc, datetime_as_string
):
    """Test writing/reading a column with a datetime far in the past.
    Dates from before 1678-1-1 aren't parsed correctly by pandas < 3.0, so they
    stay strings.
    Reported in https://github.com/geopandas/pyogrio/issues/553.
    """
    handler = contextlib.nullcontext()
    overflow_occured = False
    if not datetime_as_string and not PANDAS_GE_30 and (not use_arrow or GDAL_GE_311):
        # When datetimes should not be returned as string and arrow is not used or
        # arrow is used with GDAL >= 3.11, `pandas.to_datetime` is used to parse the
        # datetimes. However, when using pandas < 3.0, this raises an
        # "Out of bounds nanosecond timestamp" error for very old dates.
        # As a result, `read_dataframe` gives a warning and the datetimes stay strings.
        handler = pytest.warns(
            UserWarning, match="Error parsing datetimes, original strings are returned"
        )
        overflow_occured = True
        # XFAIL: datetimes before 1678-1-1 give overflow with arrow=False and pandas<3.0
    elif use_arrow and not PANDAS_GE_20 and not GDAL_GE_311:
        # When arrow is used with pandas < 2.0 and GDAL < 3.11, an overflow occurs in
        # pyarrow.to_pandas().
        handler = pytest.raises(
            Exception,
            match=re.escape("Casting from timestamp[ms] to timestamp[ns] would result"),
        )
        overflow_occured = True
        # XFAIL: datetimes before 1678-1-1 give overflow with arrow=True and pandas<2.0

    with handler:
        df = read_dataframe(
            geojson_datetime_long_ago,
            use_arrow=use_arrow,
            datetime_as_string=datetime_as_string,
            mixed_offsets_as_utc=mixed_offsets_as_utc,
        )

        exp_dates_str = pd.Series(["1670-01-01T09:00:00"], name="datetime_col")
        if datetime_as_string:
            assert is_string_dtype(df.datetime_col.dtype)
            assert_series_equal(df.datetime_col, exp_dates_str)
        else:
            # It is a single naive datetime, so regardless of mixed_offsets_as_utc the
            # expected "ideal" result is the same: a datetime64 without time zone info.
            if overflow_occured:
                # Strings are returned because of an overflow.
                assert is_string_dtype(df.datetime_col.dtype)
                assert_series_equal(df.datetime_col, exp_dates_str)
            else:
                # With use_arrow or pandas >= 3.0, old datetimes are parsed correctly.
                assert is_datetime64_dtype(df.datetime_col)
                assert df.datetime_col.iloc[0] == pd.Timestamp(1670, 1, 1, 9, 0, 0)
                assert df.datetime_col.iloc[0].unit == "ms"


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext != ".shp"])
@pytest.mark.parametrize("datetime_as_string", [False, True])
@pytest.mark.parametrize("mixed_offsets_as_utc", [False, True])
@pytest.mark.requires_arrow_write_api
def test_write_read_datetime_no_tz(
    tmp_path, ext, datetime_as_string, mixed_offsets_as_utc, use_arrow
):
    """Test writing/reading a column with naive datetimes (no time zone information)."""
    dates_raw = ["2020-01-01T09:00:00.123", "2020-01-01T10:00:00", np.nan]
    if PANDAS_GE_20:
        dates = pd.to_datetime(dates_raw, format="ISO8601").as_unit("ms")
    else:
        dates = pd.to_datetime(dates_raw)
    df = gp.GeoDataFrame(
        {"dates": dates, "geometry": [Point(1, 1)] * 3}, crs="EPSG:4326"
    )

    fpath = tmp_path / f"test{ext}"
    write_dataframe(df, fpath, use_arrow=use_arrow)
    result = read_dataframe(
        fpath,
        use_arrow=use_arrow,
        datetime_as_string=datetime_as_string,
        mixed_offsets_as_utc=mixed_offsets_as_utc,
    )

    if use_arrow and ext == ".gpkg" and __gdal_version__ < (3, 11, 0):
        # With GDAL < 3.11 with arrow, columns with naive datetimes are written
        # correctly, but when read they are wrongly interpreted as being in UTC.
        # The reason is complicated, but more info can be found e.g. here:
        # https://github.com/geopandas/pyogrio/issues/487#issuecomment-2423762807
        exp_dates = df.dates.dt.tz_localize("UTC")
        if datetime_as_string:
            exp_dates = exp_dates.astype("str").str.replace(" ", "T")
            exp_dates[2] = np.nan
            assert_series_equal(result.dates, exp_dates)
        elif not mixed_offsets_as_utc:
            assert_series_equal(result.dates, exp_dates)
        # XFAIL: naive datetimes read wrong in GPKG with GDAL < 3.11 via arrow

    elif datetime_as_string:
        assert is_string_dtype(result.dates.dtype)
        if use_arrow and __gdal_version__ < (3, 11, 0):
            dates_str = df.dates.astype("str").str.replace(" ", "T")
            dates_str[2] = np.nan
        else:
            dates_str = pd.Series(dates_raw, name="dates")
        assert_series_equal(result.dates, dates_str)
    else:
        assert is_datetime64_dtype(result.dates.dtype)
        assert_geodataframe_equal(result, df)


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext != ".shp"])
@pytest.mark.parametrize("datetime_as_string", [False, True])
@pytest.mark.parametrize("mixed_offsets_as_utc", [False, True])
@pytest.mark.filterwarnings("ignore: Non-conformant content for record 1 in column ")
@pytest.mark.requires_arrow_write_api
def test_write_read_datetime_tz(
    request, tmp_path, ext, datetime_as_string, mixed_offsets_as_utc, use_arrow
):
    """Write and read file with all equal time zones.

    This should result in the result being in pandas datetime64 dtype column.
    """
    if use_arrow and __gdal_version__ < (3, 10, 0) and ext in (".geojson", ".geojsonl"):
        # With GDAL < 3.10 with arrow, the time zone offset was applied to the datetime
        # as well as retaining the time zone.
        # This was fixed in https://github.com/OSGeo/gdal/pull/11049
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Wrong datetimes read in GeoJSON with GDAL < 3.10 via arrow"
            )
        )

    dates_raw = ["2020-01-01T09:00:00.123-05:00", "2020-01-01T10:00:00-05:00", np.nan]
    if PANDAS_GE_20:
        dates = pd.to_datetime(dates_raw, format="ISO8601").as_unit("ms")
    else:
        dates = pd.to_datetime(dates_raw)

    # Make the index non-consecutive to test this case as well. Added for issue
    # https://github.com/geopandas/pyogrio/issues/324
    df = gp.GeoDataFrame(
        {"dates": dates, "geometry": [Point(1, 1)] * 3},
        index=[0, 2, 3],
        crs="EPSG:4326",
    )
    assert isinstance(df.dates.dtype, pd.DatetimeTZDtype)

    fpath = tmp_path / f"test{ext}"
    write_dataframe(df, fpath, use_arrow=use_arrow)
    result = read_dataframe(
        fpath,
        use_arrow=use_arrow,
        datetime_as_string=datetime_as_string,
        mixed_offsets_as_utc=mixed_offsets_as_utc,
    )

    # With some older versions, the offset is represented slightly differently
    if result.dates.dtype.name.endswith(", pytz.FixedOffset(-300)]"):
        result.dates = result.dates.astype(df.dates.dtype)

    if use_arrow and ext in (".fgb", ".gpkg") and __gdal_version__ < (3, 11, 0):
        # With GDAL < 3.11 with arrow, datetime columns are written as string type
        df_exp = df.copy()
        df_exp.dates = df_exp[df_exp.dates.notna()].dates.astype(str)
        assert_series_equal(result.dates, df_exp.dates, check_index=False)
        # XFAIL: datetime columns written as string with GDAL < 3.11 via arrow
    elif datetime_as_string:
        assert is_string_dtype(result.dates.dtype)
        if use_arrow and __gdal_version__ < (3, 11, 0):
            dates_str = df.dates.astype("str").str.replace(" ", "T")
            dates_str.iloc[2] = np.nan
        elif __gdal_version__ < (3, 7, 0):
            # With GDAL < 3.7, time zone minutes aren't included in the string
            dates_str = [x[:-3] for x in dates_raw if pd.notna(x)] + [np.nan]
            dates_str = pd.Series(dates_str, name="dates")
        else:
            dates_str = pd.Series(dates_raw, name="dates")
        assert_series_equal(result.dates, dates_str, check_index=False)
    else:
        assert_series_equal(result.dates, df.dates, check_index=False)


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext != ".shp"])
@pytest.mark.parametrize("datetime_as_string", [False, True])
@pytest.mark.parametrize("mixed_offsets_as_utc", [False, True])
@pytest.mark.filterwarnings(
    "ignore: Non-conformant content for record 1 in column dates"
)
@pytest.mark.requires_arrow_write_api
def test_write_read_datetime_tz_localized_mixed_offset(
    tmp_path, ext, datetime_as_string, mixed_offsets_as_utc, use_arrow
):
    """Test with localized dates across a different summer/winter time zone offset."""
    # Australian Summer Time AEDT (GMT+11), Standard Time AEST (GMT+10)
    dates_raw = ["2023-01-01 11:00:01.111", "2023-06-01 10:00:01.111", np.nan]
    dates_naive = pd.Series(pd.to_datetime(dates_raw), name="dates")
    dates_local = dates_naive.dt.tz_localize("Australia/Sydney")
    dates_local_offsets_str = dates_local.astype(str)
    if datetime_as_string:
        exp_dates = dates_local_offsets_str.str.replace(" ", "T")
        exp_dates = exp_dates.str.replace(".111000", ".111")
        if __gdal_version__ < (3, 7, 0):
            # With GDAL < 3.7, time zone minutes aren't included in the string
            exp_dates = exp_dates.str.slice(0, -3)
    elif mixed_offsets_as_utc:
        exp_dates = dates_local.dt.tz_convert("UTC")
        if PANDAS_GE_20:
            exp_dates = exp_dates.dt.as_unit("ms")
    else:
        exp_dates = dates_local_offsets_str.apply(
            lambda x: pd.Timestamp(x) if pd.notna(x) else None
        )

    df = gp.GeoDataFrame(
        {"dates": dates_local, "geometry": [Point(1, 1)] * 3}, crs="EPSG:4326"
    )
    fpath = tmp_path / f"test{ext}"
    write_dataframe(df, fpath, use_arrow=use_arrow)
    result = read_dataframe(
        fpath,
        use_arrow=use_arrow,
        datetime_as_string=datetime_as_string,
        mixed_offsets_as_utc=mixed_offsets_as_utc,
    )

    if use_arrow and __gdal_version__ < (3, 11, 0):
        if ext in (".geojson", ".geojsonl"):
            # With GDAL < 3.11 with arrow, GDAL converts mixed time zone datetimes to
            # UTC when read as the arrow datetime column type does not support mixed tz.
            dates_utc = dates_local.dt.tz_convert("UTC")
            if PANDAS_GE_20:
                dates_utc = dates_utc.dt.as_unit("ms")
            if datetime_as_string:
                assert is_string_dtype(result.dates.dtype)
                dates_utc = dates_utc.astype(str).str.replace(" ", "T")
            assert pd.isna(result.dates[2])
            assert_series_equal(result.dates.head(2), dates_utc.head(2))
            # XFAIL: mixed tz datetimes converted to UTC with GDAL < 3.11 + arrow
            return

        elif ext in (".gpkg", ".fgb"):
            # With GDAL < 3.11 with arrow, datetime columns written as string type
            assert pd.isna(result.dates[2])
            assert_series_equal(result.dates.head(2), dates_local_offsets_str.head(2))
            # XFAIL: datetime columns written as string with GDAL < 3.11 + arrow
            return

    # GDAL tz only encodes offsets, not time zones
    if datetime_as_string:
        assert is_string_dtype(result.dates.dtype)
    elif mixed_offsets_as_utc:
        assert isinstance(result.dates.dtype, pd.DatetimeTZDtype)
    else:
        assert is_object_dtype(result.dates.dtype)

    # Check isna for the third value seperately as depending on versions this is
    # different + pandas 3.0 assert_series_equal becomes strict about this.
    assert pd.isna(result.dates[2])
    assert_series_equal(result.dates.head(2), exp_dates.head(2))


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext != ".shp"])
@pytest.mark.parametrize("datetime_as_string", [False, True])
@pytest.mark.parametrize("mixed_offsets_as_utc", [False, True])
@pytest.mark.filterwarnings(
    "ignore: Non-conformant content for record 1 in column dates"
)
@pytest.mark.requires_arrow_write_api
def test_write_read_datetime_tz_mixed_offsets(
    tmp_path, ext, datetime_as_string, mixed_offsets_as_utc, use_arrow
):
    """Test with dates with mixed time zone offsets."""
    # Pandas datetime64 column types doesn't support mixed time zone offsets, so
    # it needs to be a list of pandas.Timestamp objects instead.
    dates = [
        pd.Timestamp("2023-01-01 11:00:01.111+01:00"),
        pd.Timestamp("2023-06-01 10:00:01.111+05:00"),
        np.nan,
    ]

    df = gp.GeoDataFrame(
        {"dates": dates, "geometry": [Point(1, 1)] * 3}, crs="EPSG:4326"
    )
    fpath = tmp_path / f"test{ext}"
    write_dataframe(df, fpath, use_arrow=use_arrow)
    result = read_dataframe(
        fpath,
        use_arrow=use_arrow,
        datetime_as_string=datetime_as_string,
        mixed_offsets_as_utc=mixed_offsets_as_utc,
    )

    if use_arrow and __gdal_version__ < (3, 11, 0):
        if ext in (".geojson", ".geojsonl"):
            # With GDAL < 3.11 with arrow, GDAL converts mixed time zone datetimes to
            # UTC when read as the arrow datetime column type does not support mixed tz.
            df_exp = df.copy()
            df_exp.dates = pd.to_datetime(dates, utc=True)
            if PANDAS_GE_20:
                df_exp.dates = df_exp.dates.dt.as_unit("ms")
            if datetime_as_string:
                df_exp.dates = df_exp.dates.astype("str").str.replace(" ", "T")
            df_exp.loc[2, "dates"] = pd.NA
            assert_geodataframe_equal(result, df_exp)
            # XFAIL: mixed tz datetimes converted to UTC with GDAL < 3.11 + arrow
            return

        elif ext in (".gpkg", ".fgb"):
            # With arrow and GDAL < 3.11, mixed time zone datetimes are written as
            # string type columns, so no proper roundtrip possible.
            df_exp = df.copy()
            df_exp.dates = df_exp.dates.astype("string").astype("O")
            assert_geodataframe_equal(result, df_exp)
            # XFAIL: mixed tz datetimes converted to UTC with GDAL < 3.11 + arrow
            return

    if datetime_as_string:
        assert is_string_dtype(result.dates.dtype)
        dates_str = df.dates.map(
            lambda x: x.isoformat(timespec="milliseconds") if pd.notna(x) else np.nan
        )
        if __gdal_version__ < (3, 7, 0):
            # With GDAL < 3.7, time zone minutes aren't included in the string
            dates_str = dates_str.str.slice(0, -3)
        assert_series_equal(result.dates, dates_str)
    elif mixed_offsets_as_utc:
        assert isinstance(result.dates.dtype, pd.DatetimeTZDtype)
        exp_dates = pd.to_datetime(df.dates, utc=True)
        if PANDAS_GE_20:
            exp_dates = exp_dates.dt.as_unit("ms")
        assert_series_equal(result.dates, exp_dates)
    else:
        assert is_object_dtype(result.dates.dtype)
        assert_geodataframe_equal(result, df)


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext != ".shp"])
@pytest.mark.parametrize(
    "dates_raw",
    [
        (
            pd.Timestamp("2020-01-01T09:00:00.123-05:00"),
            pd.Timestamp("2020-01-01T10:00:00-05:00"),
            np.nan,
        ),
        (
            datetime.fromisoformat("2020-01-01T09:00:00.123-05:00"),
            datetime.fromisoformat("2020-01-01T10:00:00-05:00"),
            np.nan,
        ),
    ],
)
@pytest.mark.parametrize("datetime_as_string", [False, True])
@pytest.mark.parametrize("mixed_offsets_as_utc", [False, True])
@pytest.mark.filterwarnings(
    "ignore: Non-conformant content for record 1 in column dates"
)
@pytest.mark.requires_arrow_write_api
def test_write_read_datetime_tz_objects(
    tmp_path, dates_raw, ext, use_arrow, datetime_as_string, mixed_offsets_as_utc
):
    """Datetime objects with equal offsets are read as datetime64."""
    dates = pd.Series(dates_raw, dtype="O")
    df = gp.GeoDataFrame(
        {"dates": dates, "geometry": [Point(1, 1)] * 3}, crs="EPSG:4326"
    )

    fpath = tmp_path / f"test{ext}"
    write_dataframe(df, fpath, use_arrow=use_arrow)
    result = read_dataframe(
        fpath,
        use_arrow=use_arrow,
        datetime_as_string=datetime_as_string,
        mixed_offsets_as_utc=mixed_offsets_as_utc,
    )

    # Check result
    if PANDAS_GE_20:
        exp_dates = pd.to_datetime(dates_raw, format="ISO8601").as_unit("ms")
    else:
        exp_dates = pd.to_datetime(dates_raw)
    exp_df = df.copy()
    exp_df["dates"] = pd.Series(exp_dates, name="dates")

    # With some older versions, the offset is represented slightly differently
    if result.dates.dtype.name.endswith(", pytz.FixedOffset(-300)]"):
        result["dates"] = result.dates.astype(exp_df.dates.dtype)

    if use_arrow and __gdal_version__ < (3, 10, 0) and ext in (".geojson", ".geojsonl"):
        # XFAIL: Wrong datetimes read in GeoJSON with GDAL < 3.10 via arrow.
        # The time zone offset was applied to the datetime as well as retaining
        # the time zone. This was fixed in https://github.com/OSGeo/gdal/pull/11049

        # Subtract 5 hours from the expected datetimes to match the wrong result.
        if datetime_as_string:
            exp_df["dates"] = pd.Series(
                [
                    "2020-01-01T04:00:00.123000-05:00",
                    "2020-01-01T05:00:00-05:00",
                    np.nan,
                ]
            )
        else:
            exp_df["dates"] = exp_df.dates - pd.Timedelta(hours=5)
            if PANDAS_GE_20:
                # The unit needs to be applied again apparently
                exp_df["dates"] = exp_df.dates.dt.as_unit("ms")
        assert_geodataframe_equal(result, exp_df)
        return

    if use_arrow and __gdal_version__ < (3, 11, 0) and ext in (".fgb", ".gpkg"):
        # XFAIL: datetime columns are written as string with GDAL < 3.11 + arrow
        # -> custom formatting because the df column is object dtype and thus
        # astype(str) converted the datetime objects one by one
        exp_df["dates"] = pd.Series(
            ["2020-01-01 09:00:00.123000-05:00", "2020-01-01 10:00:00-05:00", np.nan]
        )
        assert_geodataframe_equal(result, exp_df)
        return

    if datetime_as_string:
        assert is_string_dtype(result.dates.dtype)
        if use_arrow and __gdal_version__ < (3, 11, 0):
            # With GDAL < 3.11 with arrow, datetime columns are written as string type
            exp_df["dates"] = pd.Series(
                [
                    "2020-01-01T09:00:00.123000-05:00",
                    "2020-01-01T10:00:00-05:00",
                    np.nan,
                ]
            )
        else:
            exp_df["dates"] = pd.Series(
                ["2020-01-01T09:00:00.123-05:00", "2020-01-01T10:00:00-05:00", np.nan]
            )
            if __gdal_version__ < (3, 7, 0):
                # With GDAL < 3.7, time zone minutes aren't included in the string
                exp_df["dates"] = exp_df.dates.str.slice(0, -3)
    elif mixed_offsets_as_utc:
        # the offsets are all -05:00, so the result retains the offset and not UTC
        assert isinstance(result.dates.dtype, pd.DatetimeTZDtype)
        assert str(result.dates.dtype.tz) in ("UTC-05:00", "pytz.FixedOffset(-300)")
    else:
        assert isinstance(result.dates.dtype, pd.DatetimeTZDtype)

    assert_geodataframe_equal(result, exp_df)


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext != ".shp"])
@pytest.mark.parametrize("datetime_as_string", [False, True])
@pytest.mark.parametrize("mixed_offsets_as_utc", [False, True])
@pytest.mark.requires_arrow_write_api
def test_write_read_datetime_utc(
    tmp_path, ext, use_arrow, datetime_as_string, mixed_offsets_as_utc
):
    """Test writing/reading a column with UTC datetimes."""
    dates_raw = ["2020-01-01T09:00:00.123Z", "2020-01-01T10:00:00Z", np.nan]
    if PANDAS_GE_20:
        dates = pd.to_datetime(dates_raw, format="ISO8601").as_unit("ms")
    else:
        dates = pd.to_datetime(dates_raw)
    df = gp.GeoDataFrame(
        {"dates": dates, "geometry": [Point(1, 1)] * 3}, crs="EPSG:4326"
    )
    assert df.dates.dtype.name in ("datetime64[ms, UTC]", "datetime64[ns, UTC]")

    fpath = tmp_path / f"test{ext}"
    write_dataframe(df, fpath, use_arrow=use_arrow)
    result = read_dataframe(
        fpath,
        use_arrow=use_arrow,
        datetime_as_string=datetime_as_string,
        mixed_offsets_as_utc=mixed_offsets_as_utc,
    )

    if use_arrow and ext == ".fgb" and __gdal_version__ < (3, 11, 0):
        # With GDAL < 3.11 with arrow, time zone information is dropped when reading
        # .fgb
        if datetime_as_string:
            assert is_string_dtype(result.dates.dtype)
            dates_str = pd.Series(
                ["2020-01-01T09:00:00.123", "2020-01-01T10:00:00.000", np.nan],
                name="dates",
            )
            assert_series_equal(result.dates, dates_str)
        else:
            assert_series_equal(result.dates, df.dates.dt.tz_localize(None))
        # XFAIL: UTC datetimes read wrong in .fgb with GDAL < 3.11 via arrow
    elif datetime_as_string:
        assert is_string_dtype(result.dates.dtype)
        if use_arrow and __gdal_version__ < (3, 11, 0):
            dates_str = df.dates.astype("str").str.replace(" ", "T")
            dates_str[2] = np.nan
        else:
            dates_str = pd.Series(dates_raw, name="dates")
            if __gdal_version__ < (3, 7, 0):
                # With GDAL < 3.7, datetime ends with +00 for UTC, not Z
                dates_str = dates_str.str.replace("Z", "+00")
        assert_series_equal(result.dates, dates_str)
    else:
        assert result.dates.dtype.name in ("datetime64[ms, UTC]", "datetime64[ns, UTC]")
        assert_geodataframe_equal(result, df)


def test_read_null_values(tmp_path, use_arrow):
    filename = tmp_path / "test_null_values_no_geometry.gpkg"

    # create a GPKG with no geometries and only null values
    expected = pd.DataFrame({"col": [None, None]})
    write_dataframe(expected, filename)

    df = read_dataframe(filename, use_arrow=use_arrow, read_geometry=False)

    # make sure that Null values are preserved
    assert df["col"].isna().all()


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
    fids_expected = pd.Index([2, 3], name="fid")
    fids_expected += START_FID[naturalearth_lowres_all_ext.suffix]
    assert_index_equal(df.index, fids_expected)


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

    if naturalearth_lowres_all_ext.suffix == ".gpkg" and __gdal_version__ >= (3, 11, 0):
        with pytest.raises(DataLayerError, match="no such column"):
            read_dataframe(
                naturalearth_lowres_all_ext, use_arrow=use_arrow, where="invalid"
            )
    else:
        with pytest.raises(ValueError, match="Invalid SQL"):
            read_dataframe(
                naturalearth_lowres_all_ext, use_arrow=use_arrow, where="invalid"
            )


def test_read_where_ignored_field(naturalearth_lowres, use_arrow):
    # column included in where is not also included in list of columns, which means
    # GDAL will return no features
    # NOTE: this behavior is inconsistent across drivers so only shapefiles are
    # tested for this
    df = read_dataframe(
        naturalearth_lowres,
        where=""" "iso_a3" = 'CAN' """,
        columns=["name"],
        use_arrow=use_arrow,
    )

    assert len(df) == 0


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
        and naturalearth_lowres_all_ext.suffix == ".gpkg"
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
        and naturalearth_lowres_all_ext.suffix == ".gpkg"
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


@pytest.mark.parametrize("fids", [[1, 5, 10], np.array([1, 5, 10], dtype=np.int64)])
def test_read_fids(naturalearth_lowres_all_ext, fids, use_arrow):
    # ensure keyword is properly passed through
    df = read_dataframe(
        naturalearth_lowres_all_ext, fids=fids, fid_as_index=True, use_arrow=use_arrow
    )
    assert len(df) == 3
    assert np.array_equal(fids, df.index.values)


@requires_pyarrow_api
def test_read_fids_arrow_max_exception(naturalearth_lowres):
    # Maximum number at time of writing is 4997 for "OGRSQL". For e.g. for SQLite based
    # formats like Geopackage, there is no limit.
    nb_fids = 4998
    fids = range(nb_fids)
    with pytest.raises(ValueError, match=f"error applying filter for {nb_fids} fids"):
        _ = read_dataframe(naturalearth_lowres, fids=fids, use_arrow=True)


@requires_pyarrow_api
@pytest.mark.skipif(
    __gdal_version__ >= (3, 8, 0), reason="GDAL >= 3.8.0 does not need to warn"
)
def test_read_fids_arrow_warning_old_gdal(naturalearth_lowres_all_ext):
    # A warning should be given for old GDAL versions, except for some file formats.
    if naturalearth_lowres_all_ext.suffix not in [".gpkg", ".geojson"]:
        handler = pytest.warns(
            UserWarning,
            match="Using 'fids' and 'use_arrow=True' with GDAL < 3.8 can be slow",
        )
    else:
        handler = contextlib.nullcontext()

    with handler:
        df = read_dataframe(naturalearth_lowres_all_ext, fids=[22], use_arrow=True)
        assert len(df) == 1


def test_read_fids_force_2d(tmp_path):
    filename = tmp_path / "test.gpkg"

    # create a GPKG with 3D point values
    expected = gp.GeoDataFrame(
        geometry=[Point(0, 0, 0), Point(1, 1, 0)], crs="EPSG:4326"
    )
    write_dataframe(expected, filename)

    df = read_dataframe(filename, fids=[1])
    assert_geodataframe_equal(df, expected.iloc[:1])

    df = read_dataframe(filename, force_2d=True, fids=[1])
    assert np.array_equal(
        df.geometry.values, shapely.force_2d(expected.iloc[:1].geometry.values)
    )


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

    if skip_features == 200 and not use_arrow:
        # result is an empty dataframe, so no proper dtype inference happens
        # for the numpy object dtype arrays
        df[["continent", "name", "iso_a3"]] = df[
            ["continent", "name", "iso_a3"]
        ].astype("str")

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


@pytest.mark.parametrize("skip_features", [0, 10, 200])
@pytest.mark.parametrize("max_features", [10, 100])
def test_read_max_features(
    naturalearth_lowres_all_ext, use_arrow, max_features, skip_features
):
    ext = naturalearth_lowres_all_ext.suffix
    expected = (
        read_dataframe(naturalearth_lowres_all_ext)
        .iloc[skip_features : skip_features + max_features]
        .reset_index(drop=True)
    )
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        skip_features=skip_features,
        max_features=max_features,
        use_arrow=use_arrow,
    )

    assert len(df) == len(expected)

    # Coordinates are not precisely equal when written to JSON
    # dtypes do not necessarily round-trip precisely through JSON
    is_json = ext in [".geojson", ".geojsonl"]
    # In .geojsonl the vertices are reordered, so normalize
    is_jsons = ext == ".geojsonl"

    if len(expected) == 0 and not use_arrow:
        # for pandas >= 3, the column has string dtype but when reading it as
        # empty result, it gets inferred as object dtype
        expected["continent"] = expected["continent"].astype("object")
        expected["name"] = expected["name"].astype("object")
        expected["iso_a3"] = expected["iso_a3"].astype("object")

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
    df = read_dataframe(
        naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL", use_arrow=use_arrow
    )
    assert len(df.columns) == 4
    assert len(df) == 177

    # Should return single row
    sql = "SELECT * FROM naturalearth_lowres WHERE iso_a3 = 'CAN'"
    df = read_dataframe(
        naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL", use_arrow=use_arrow
    )
    assert len(df) == 1
    assert len(df.columns) == 6
    assert df.iloc[0].iso_a3 == "CAN"

    sql = """SELECT *
               FROM naturalearth_lowres
              WHERE iso_a3 IN ('CAN', 'USA', 'MEX')"""
    df = read_dataframe(
        naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL", use_arrow=use_arrow
    )
    assert len(df.columns) == 6
    assert len(df) == 3
    assert df.iso_a3.tolist() == ["CAN", "USA", "MEX"]

    sql = """SELECT *
               FROM naturalearth_lowres
              WHERE iso_a3 IN ('CAN', 'USA', 'MEX')
              ORDER BY name"""
    df = read_dataframe(
        naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL", use_arrow=use_arrow
    )
    assert len(df.columns) == 6
    assert len(df) == 3
    assert df.iso_a3.tolist() == ["CAN", "MEX", "USA"]

    # Should return items within range.
    sql = """SELECT *
               FROM naturalearth_lowres
              WHERE POP_EST >= 10000000 AND POP_EST < 100000000"""
    df = read_dataframe(
        naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL", use_arrow=use_arrow
    )
    assert len(df) == 75
    assert len(df.columns) == 6
    assert df.pop_est.min() >= 10000000
    assert df.pop_est.max() < 100000000

    # Should match no items.
    sql = "SELECT * FROM naturalearth_lowres WHERE ISO_A3 = 'INVALID'"
    df = read_dataframe(
        naturalearth_lowres_all_ext, sql=sql, sql_dialect="OGRSQL", use_arrow=use_arrow
    )
    assert len(df) == 0


def test_read_sql_invalid(naturalearth_lowres_all_ext, use_arrow):
    if naturalearth_lowres_all_ext.suffix == ".gpkg":
        with pytest.raises(Exception, match="In ExecuteSQL().*"):
            read_dataframe(
                naturalearth_lowres_all_ext, sql="invalid", use_arrow=use_arrow
            )
    else:
        with pytest.raises(Exception, match="SQL Expression Parsing Error"):
            read_dataframe(
                naturalearth_lowres_all_ext, sql="invalid", use_arrow=use_arrow
            )

    with pytest.raises(
        ValueError, match="'sql' parameter cannot be combined with 'layer'"
    ):
        read_dataframe(
            naturalearth_lowres_all_ext,
            sql="whatever",
            layer="invalid",
            use_arrow=use_arrow,
        )


def test_read_sql_columns_where(naturalearth_lowres_all_ext, use_arrow):
    sql = "SELECT iso_a3 AS iso_a3_renamed, name, pop_est FROM naturalearth_lowres"
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        sql=sql,
        sql_dialect="OGRSQL",
        columns=["iso_a3_renamed", "name"],
        where="iso_a3_renamed IN ('CAN', 'USA', 'MEX')",
        use_arrow=use_arrow,
    )
    assert len(df.columns) == 3
    assert len(df) == 3
    assert df.iso_a3_renamed.tolist() == ["CAN", "USA", "MEX"]


def test_read_sql_columns_where_bbox(naturalearth_lowres_all_ext, use_arrow):
    sql = "SELECT iso_a3 AS iso_a3_renamed, name, pop_est FROM naturalearth_lowres"
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        sql=sql,
        sql_dialect="OGRSQL",
        columns=["iso_a3_renamed", "name"],
        where="iso_a3_renamed IN ('CRI', 'PAN')",
        bbox=(-85, 8, -80, 10),
        use_arrow=use_arrow,
    )
    assert len(df.columns) == 3
    assert len(df) == 2
    assert df.iso_a3_renamed.tolist() == ["PAN", "CRI"]


def test_read_sql_skip_max(naturalearth_lowres_all_ext, use_arrow):
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
        use_arrow=use_arrow,
    )
    assert len(df.columns) == 6
    assert len(df) == 1
    assert df.iso_a3.tolist() == ["MEX"]

    sql = "SELECT * FROM naturalearth_lowres LIMIT 1"
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        sql=sql,
        max_features=3,
        sql_dialect="OGRSQL",
        use_arrow=use_arrow,
    )
    assert len(df) == 1

    sql = "SELECT * FROM naturalearth_lowres LIMIT 1"
    df = read_dataframe(
        naturalearth_lowres_all_ext,
        sql=sql,
        sql_dialect="OGRSQL",
        skip_features=1,
        use_arrow=use_arrow,
    )
    assert len(df) == 0


@requires_gdal_geos
@pytest.mark.parametrize(
    "naturalearth_lowres",
    [ext for ext in ALL_EXTS if ext != ".gpkg"],
    indirect=["naturalearth_lowres"],
)
def test_read_sql_dialect_sqlite_nogpkg(naturalearth_lowres, use_arrow):
    # Should return singular item
    sql = "SELECT * FROM naturalearth_lowres WHERE iso_a3 = 'CAN'"
    df = read_dataframe(
        naturalearth_lowres, sql=sql, sql_dialect="SQLITE", use_arrow=use_arrow
    )
    assert len(df) == 1
    assert len(df.columns) == 6
    assert df.iloc[0].iso_a3 == "CAN"
    area_canada = df.iloc[0].geometry.area

    # Use spatialite function
    sql = """SELECT ST_Buffer(geometry, 5) AS geometry, name, pop_est, iso_a3
               FROM naturalearth_lowres
              WHERE ISO_A3 = 'CAN'"""
    df = read_dataframe(
        naturalearth_lowres, sql=sql, sql_dialect="SQLITE", use_arrow=use_arrow
    )
    assert len(df) == 1
    assert len(df.columns) == 4
    assert df.iloc[0].geometry.area > area_canada


@requires_gdal_geos
@pytest.mark.parametrize(
    "naturalearth_lowres", [".gpkg"], indirect=["naturalearth_lowres"]
)
def test_read_sql_dialect_sqlite_gpkg(naturalearth_lowres, use_arrow):
    # "INDIRECT_SQL" prohibits GDAL from passing the SQL statement to sqlite.
    # Because the statement is processed within GDAL it is possible to use
    # spatialite functions even if sqlite isn't built with spatialite support.
    sql = "SELECT * FROM naturalearth_lowres WHERE iso_a3 = 'CAN'"
    df = read_dataframe(
        naturalearth_lowres, sql=sql, sql_dialect="INDIRECT_SQLITE", use_arrow=use_arrow
    )
    assert len(df) == 1
    assert len(df.columns) == 6
    assert df.iloc[0].iso_a3 == "CAN"
    area_canada = df.iloc[0].geometry.area

    # Use spatialite function
    sql = """SELECT ST_Buffer(geom, 5) AS geometry, name, pop_est, iso_a3
               FROM naturalearth_lowres
              WHERE ISO_A3 = 'CAN'"""
    df = read_dataframe(
        naturalearth_lowres, sql=sql, sql_dialect="INDIRECT_SQLITE", use_arrow=use_arrow
    )
    assert len(df) == 1
    assert len(df.columns) == 4
    assert df.iloc[0].geometry.area > area_canada


@pytest.mark.parametrize(
    "encoding, arrow",
    [
        ("utf-8", False),
        pytest.param("utf-8", True, marks=requires_arrow_write_api),
        ("cp1252", False),
        (None, False),
    ],
)
def test_write_csv_encoding(tmp_path, encoding, arrow):
    """Test if write_dataframe uses the default encoding correctly.

    Arrow only supports utf-8 encoding.
    """
    # Write csv test file. Depending on the os this will be written in a different
    # encoding: for linux and macos this is utf-8, for windows it is cp1252.
    csv_path = tmp_path / "test.csv"

    with open(csv_path, "w", encoding=encoding) as csv:
        csv.write("näme,city\n")
        csv.write("Wilhelm Röntgen,Zürich\n")

    # Write csv test file with the same data using write_dataframe. It should use the
    # same encoding as above.
    df = pd.DataFrame({"näme": ["Wilhelm Röntgen"], "city": ["Zürich"]})
    csv_pyogrio_path = tmp_path / "test_pyogrio.csv"
    write_dataframe(df, csv_pyogrio_path, encoding=encoding, use_arrow=arrow)

    # Check if the text files written both ways can be read again and give same result.
    with open(csv_path, encoding=encoding) as csv:
        csv_str = csv.read()
    with open(csv_pyogrio_path, encoding=encoding) as csv_pyogrio:
        csv_pyogrio_str = csv_pyogrio.read()
    assert csv_str == csv_pyogrio_str

    # Check if they files are binary identical, to be 100% sure they were written with
    # the same encoding.
    with open(csv_path, "rb") as csv:
        csv_bytes = csv.read()
    with open(csv_pyogrio_path, "rb") as csv_pyogrio:
        csv_pyogrio_bytes = csv_pyogrio.read()
    assert csv_bytes == csv_pyogrio_bytes


@pytest.mark.parametrize(
    "ext, fid_column, fid_param_value",
    [
        (".gpkg", "fid", None),
        (".gpkg", "FID", None),
        (".sqlite", "ogc_fid", None),
        (".gpkg", "fid_custom", "fid_custom"),
        (".gpkg", "FID_custom", "fid_custom"),
        (".sqlite", "ogc_fid_custom", "ogc_fid_custom"),
    ],
)
@pytest.mark.requires_arrow_write_api
def test_write_custom_fids(tmp_path, ext, fid_column, fid_param_value, use_arrow):
    """Test to specify FIDs to save when writing to a file.

    Saving custom FIDs is only supported for formats that actually store the FID, like
    e.g. GPKG and SQLite. The fid_column name check is case-insensitive.

    Typically, GDAL supports using a custom FID column for these file formats via a
    `FID` layer creation option, which is also tested here. If `fid_param_value` is
    specified (not None), an `fid` parameter is passed to `write_dataframe`, causing
    GDAL to use the column name specified for the FID.
    """
    input_gdf = gp.GeoDataFrame(
        {fid_column: [5]}, geometry=[shapely.Point(0, 0)], crs="epsg:4326"
    )
    kwargs = {}
    if fid_param_value is not None:
        kwargs["fid"] = fid_param_value
    path = tmp_path / f"test{ext}"

    write_dataframe(input_gdf, path, use_arrow=use_arrow, **kwargs)

    assert path.exists()
    output_gdf = read_dataframe(path, fid_as_index=True, use_arrow=use_arrow)
    output_gdf = output_gdf.reset_index()

    # pyogrio always sets "fid" as index name with `fid_as_index`
    expected_gdf = input_gdf.rename(columns={fid_column: "fid"})
    assert_geodataframe_equal(output_gdf, expected_gdf)


@pytest.mark.parametrize("ext", ALL_EXTS)
@pytest.mark.requires_arrow_write_api
def test_write_dataframe(tmp_path, naturalearth_lowres, ext, use_arrow):
    input_gdf = read_dataframe(naturalearth_lowres)
    output_path = tmp_path / f"test{ext}"

    if ext == ".fgb":
        # For .fgb, spatial_index=False to avoid the rows being reordered
        write_dataframe(
            input_gdf, output_path, use_arrow=use_arrow, spatial_index=False
        )
    else:
        write_dataframe(input_gdf, output_path, use_arrow=use_arrow)

    assert output_path.exists()
    result_gdf = read_dataframe(output_path)

    geometry_types = result_gdf.geometry.type.unique()
    if DRIVERS[ext] in DRIVERS_NO_MIXED_SINGLE_MULTI:
        assert list(geometry_types) == ["MultiPolygon"]
    else:
        assert set(geometry_types) == {"MultiPolygon", "Polygon"}

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
@pytest.mark.parametrize("write_geodf", [True, False])
@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS + [".xlsx"] if ext != ".fgb"])
@pytest.mark.requires_arrow_write_api
def test_write_dataframe_no_geom(
    request, tmp_path, naturalearth_lowres, write_geodf, ext, use_arrow
):
    """Test writing a (geo)dataframe without a geometry column.

    FlatGeobuf (.fgb) doesn't seem to support this, and just writes an empty file.
    """
    # Prepare test data
    input_df = read_dataframe(naturalearth_lowres, read_geometry=False)
    if write_geodf:
        input_df = gp.GeoDataFrame(input_df)

    output_path = tmp_path / f"test{ext}"

    # A shapefile without geometry column results in only a .dbf file.
    if ext == ".shp":
        output_path = output_path.with_suffix(".dbf")

    # Determine driver
    driver = DRIVERS[ext] if ext != ".xlsx" else "XLSX"

    write_dataframe(input_df, output_path, use_arrow=use_arrow, driver=driver)

    assert output_path.exists()
    result_df = read_dataframe(output_path)

    assert isinstance(result_df, pd.DataFrame)

    # some dtypes do not round-trip precisely through these file types
    check_dtype = ext not in [".geojson", ".geojsonl", ".xlsx"]

    if ext in [".gpkg", ".shp", ".xlsx"]:
        # These file types return a DataFrame when read.
        assert not isinstance(result_df, gp.GeoDataFrame)
        if isinstance(input_df, gp.GeoDataFrame):
            input_df = pd.DataFrame(input_df)

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


@pytest.mark.requires_arrow_write_api
def test_write_dataframe_index(tmp_path, naturalearth_lowres, use_arrow):
    # dataframe writing ignores the index
    input_gdf = read_dataframe(naturalearth_lowres)
    input_gdf = input_gdf.set_index("iso_a3")

    output_path = tmp_path / "test.shp"
    write_dataframe(input_gdf, output_path, use_arrow=use_arrow)

    result_gdf = read_dataframe(output_path)
    assert isinstance(result_gdf.index, pd.RangeIndex)
    assert_geodataframe_equal(result_gdf, input_gdf.reset_index(drop=True))


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext not in ".geojsonl"])
@pytest.mark.parametrize(
    "columns, dtype",
    [
        ([], None),
        (["col_int"], np.int64),
        (["col_float"], np.float64),
        (["col_object"], object),
    ],
)
@pytest.mark.requires_arrow_write_api
def test_write_empty_dataframe(tmp_path, ext, columns, dtype, use_arrow):
    """Test writing dataframe with no rows.

    With use_arrow, object type columns with no rows are converted to null type columns
    by pyarrow, but null columns are not supported by GDAL. Added to test fix for #513.
    """
    expected = gp.GeoDataFrame(geometry=[], columns=columns, dtype=dtype, crs=4326)
    filename = tmp_path / f"test{ext}"
    write_dataframe(expected, filename, use_arrow=use_arrow)

    assert filename.exists()
    df = read_dataframe(filename, use_arrow=use_arrow)

    # Check result
    # For older pandas versions, the index is created as Object dtype but read as
    # RangeIndex, so don't check the index dtype in that case.
    check_index_type = True if PANDAS_GE_20 else False
    # with pandas 3+ and reading through arrow, we preserve the string dtype
    # (no proper dtype inference happens for the empty numpy object dtype arrays)
    if use_arrow and dtype is object:
        expected["col_object"] = expected["col_object"].astype("str")
    assert_geodataframe_equal(df, expected, check_index_type=check_index_type)


def test_write_empty_geometry(tmp_path):
    expected = gp.GeoDataFrame({"x": [0]}, geometry=from_wkt(["POINT EMPTY"]), crs=4326)
    filename = tmp_path / "test.gpkg"

    # Check that no warning is raised with GeoSeries.notna()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        if not HAS_PYPROJ:
            warnings.filterwarnings("ignore", message="'crs' was not provided.")
        write_dataframe(expected, filename)
    assert filename.exists()

    # Xref GH-436: round-tripping possible with GPKG but not others
    df = read_dataframe(filename)
    assert_geodataframe_equal(df, expected)


@pytest.mark.requires_arrow_write_api
def test_write_None_string_column(tmp_path, use_arrow):
    """Test pandas object columns with all None values.

    With use_arrow, such columns are converted to null type columns by pyarrow, but null
    columns are not supported by GDAL. Added to test fix for #513.
    """
    gdf = gp.GeoDataFrame({"object_col": [None]}, geometry=[Point(0, 0)], crs=4326)
    filename = tmp_path / "test.gpkg"

    write_dataframe(gdf, filename, use_arrow=use_arrow)
    assert filename.exists()

    result_gdf = read_dataframe(filename, use_arrow=use_arrow)
    if (
        PANDAS_GE_30 or (PANDAS_GE_23 and pd.options.future.infer_string)
    ) and use_arrow:
        assert result_gdf.object_col.dtype == "str"
        gdf["object_col"] = gdf["object_col"].astype("str")
    else:
        assert result_gdf.object_col.dtype == object
    assert_geodataframe_equal(result_gdf, gdf)


@pytest.mark.parametrize("ext", [".geojsonl", ".geojsons"])
@pytest.mark.requires_arrow_write_api
def test_write_read_empty_dataframe_unsupported(tmp_path, ext, use_arrow):
    # Writing empty dataframe to .geojsons or .geojsonl results logically in a 0 byte
    # file, but gdal isn't able to read those again at the time of writing.
    # Issue logged here: https://github.com/geopandas/pyogrio/issues/94
    expected = gp.GeoDataFrame(geometry=[], crs=4326)

    filename = tmp_path / f"test{ext}"
    write_dataframe(expected, filename, use_arrow=use_arrow)

    assert filename.exists()
    with pytest.raises(
        Exception, match=".* not recognized as( being in)? a supported file format."
    ):
        _ = read_dataframe(filename, use_arrow=use_arrow)


@pytest.mark.requires_arrow_write_api
def test_write_dataframe_gpkg_multiple_layers(tmp_path, naturalearth_lowres, use_arrow):
    input_gdf = read_dataframe(naturalearth_lowres)
    filename = tmp_path / "test.gpkg"

    write_dataframe(
        input_gdf,
        filename,
        layer="first",
        promote_to_multi=True,
        use_arrow=use_arrow,
    )

    assert filename.exists()
    assert np.array_equal(list_layers(filename), [["first", "MultiPolygon"]])

    write_dataframe(
        input_gdf,
        filename,
        layer="second",
        promote_to_multi=True,
        use_arrow=use_arrow,
    )
    assert np.array_equal(
        list_layers(filename),
        [["first", "MultiPolygon"], ["second", "MultiPolygon"]],
    )


@pytest.mark.parametrize("ext", ALL_EXTS)
@pytest.mark.requires_arrow_write_api
def test_write_dataframe_append(request, tmp_path, naturalearth_lowres, ext, use_arrow):
    if use_arrow and ext.startswith(".geojson"):
        # Bug in GDAL when appending int64 to GeoJSON
        # (https://github.com/OSGeo/gdal/issues/9792)
        request.node.add_marker(
            pytest.mark.xfail(reason="Bugs with append when writing Arrow to GeoJSON")
        )

    input_gdf = read_dataframe(naturalearth_lowres)
    filename = tmp_path / f"test{ext}"

    write_dataframe(input_gdf, filename, use_arrow=use_arrow)

    filename.exists()
    assert len(read_dataframe(filename)) == 177

    write_dataframe(input_gdf, filename, use_arrow=use_arrow, append=True)
    assert len(read_dataframe(filename)) == 354


@pytest.mark.parametrize("spatial_index", [False, True])
@pytest.mark.requires_arrow_write_api
def test_write_dataframe_gdal_options(
    tmp_path, naturalearth_lowres, spatial_index, use_arrow
):
    df = read_dataframe(naturalearth_lowres)

    outfilename1 = tmp_path / "test1.shp"
    write_dataframe(
        df,
        outfilename1,
        use_arrow=use_arrow,
        SPATIAL_INDEX="YES" if spatial_index else "NO",
    )
    assert outfilename1.exists() is True
    index_filename1 = tmp_path / "test1.qix"
    assert index_filename1.exists() is spatial_index

    # using explicit layer_options instead
    outfilename2 = tmp_path / "test2.shp"
    write_dataframe(
        df,
        outfilename2,
        use_arrow=use_arrow,
        layer_options={"spatial_index": spatial_index},
    )
    assert outfilename2.exists() is True
    index_filename2 = tmp_path / "test2.qix"
    assert index_filename2.exists() is spatial_index


@pytest.mark.requires_arrow_write_api
def test_write_dataframe_gdal_options_unknown(tmp_path, naturalearth_lowres, use_arrow):
    df = read_dataframe(naturalearth_lowres)

    # geojson has no spatial index, so passing keyword should raise
    outfilename = tmp_path / "test.geojson"
    with pytest.raises(ValueError, match="unrecognized option 'SPATIAL_INDEX'"):
        write_dataframe(df, outfilename, use_arrow=use_arrow, spatial_index=True)


def _get_gpkg_table_names(path):
    import sqlite3

    con = sqlite3.connect(path)
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    result = cursor.fetchall()
    return [res[0] for res in result]


@pytest.mark.requires_arrow_write_api
def test_write_dataframe_gdal_options_dataset(tmp_path, naturalearth_lowres, use_arrow):
    df = read_dataframe(naturalearth_lowres)

    test_default_filename = tmp_path / "test_default.gpkg"
    write_dataframe(df, test_default_filename, use_arrow=use_arrow)
    assert "gpkg_ogr_contents" in _get_gpkg_table_names(test_default_filename)

    test_no_contents_filename = tmp_path / "test_no_contents.gpkg"
    write_dataframe(
        df, test_default_filename, use_arrow=use_arrow, ADD_GPKG_OGR_CONTENTS="NO"
    )
    assert "gpkg_ogr_contents" not in _get_gpkg_table_names(test_no_contents_filename)

    test_no_contents_filename2 = tmp_path / "test_no_contents2.gpkg"
    write_dataframe(
        df,
        test_no_contents_filename2,
        use_arrow=use_arrow,
        dataset_options={"add_gpkg_ogr_contents": False},
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
@pytest.mark.requires_arrow_write_api
def test_write_dataframe_promote_to_multi(
    tmp_path,
    naturalearth_lowres,
    ext,
    promote_to_multi,
    expected_geometry_types,
    expected_geometry_type,
    use_arrow,
):
    input_gdf = read_dataframe(naturalearth_lowres)

    output_path = tmp_path / f"test_promote{ext}"
    write_dataframe(
        input_gdf, output_path, use_arrow=use_arrow, promote_to_multi=promote_to_multi
    )

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
@pytest.mark.requires_arrow_write_api
def test_write_dataframe_promote_to_multi_layer_geom_type(
    tmp_path,
    naturalearth_lowres,
    ext,
    promote_to_multi,
    geometry_type,
    expected_geometry_types,
    expected_geometry_type,
    use_arrow,
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
            use_arrow=use_arrow,
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
        (
            ".shp",
            None,
            "Point",
            "Could not add feature to layer at index|Error while writing batch to OGR "
            "layer",
        ),
    ],
)
@pytest.mark.requires_arrow_write_api
def test_write_dataframe_promote_to_multi_layer_geom_type_invalid(
    tmp_path,
    naturalearth_lowres,
    ext,
    promote_to_multi,
    geometry_type,
    expected_raises_match,
    use_arrow,
):
    input_gdf = read_dataframe(naturalearth_lowres)

    output_path = tmp_path / f"test{ext}"
    with pytest.raises((FeatureError, DataLayerError), match=expected_raises_match):
        write_dataframe(
            input_gdf,
            output_path,
            use_arrow=use_arrow,
            promote_to_multi=promote_to_multi,
            geometry_type=geometry_type,
        )


@pytest.mark.requires_arrow_write_api
def test_write_dataframe_layer_geom_type_invalid(
    tmp_path, naturalearth_lowres, use_arrow
):
    df = read_dataframe(naturalearth_lowres)

    filename = tmp_path / "test.geojson"
    with pytest.raises(
        GeometryError, match="Geometry type is not supported: NotSupported"
    ):
        write_dataframe(df, filename, use_arrow=use_arrow, geometry_type="NotSupported")


@pytest.mark.parametrize("ext", [ext for ext in ALL_EXTS if ext not in ".shp"])
@pytest.mark.requires_arrow_write_api
def test_write_dataframe_truly_mixed(tmp_path, ext, use_arrow):
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
        write_dataframe(df, filename, use_arrow=use_arrow, spatial_index=False)
    else:
        write_dataframe(df, filename, use_arrow=use_arrow)

    # Drivers that support mixed geometries will default to "Unknown" geometry type
    assert read_info(filename)["geometry_type"] == "Unknown"
    result = read_dataframe(filename)
    assert_geodataframe_equal(result, df, check_geom_type=True)


@pytest.mark.requires_arrow_write_api
def test_write_dataframe_truly_mixed_invalid(tmp_path, use_arrow):
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
        # DataLayerError when using Arrow
        "|Error while writing batch to OGR layer: Attempt to "
        r"write non-point \(LINESTRING\) geometry to point shapefile."
    )
    with pytest.raises((FeatureError, DataLayerError), match=msg):
        write_dataframe(df, tmp_path / "test.shp", use_arrow=use_arrow)


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
@pytest.mark.requires_arrow_write_api
def test_write_dataframe_infer_geometry_with_nulls(tmp_path, geoms, ext, use_arrow):
    filename = tmp_path / f"test{ext}"

    df = gp.GeoDataFrame({"col": [1.0, 2.0]}, geometry=geoms, crs="EPSG:4326")
    write_dataframe(df, filename, use_arrow=use_arrow)
    result = read_dataframe(filename)
    assert_geodataframe_equal(result, df)


@pytest.mark.filterwarnings(
    "ignore: You will likely lose important projection information"
)
@pytest.mark.requires_arrow_write_api
@requires_pyproj
def test_custom_crs_io(tmp_path, naturalearth_lowres_all_ext, use_arrow):
    df = read_dataframe(naturalearth_lowres_all_ext)
    # project Belgium to a custom Albers Equal Area projection
    expected = (
        df.loc[df.name == "Belgium"]
        .reset_index(drop=True)
        .to_crs("+proj=aea +lat_1=49.5 +lat_2=51.5 +lon_0=4.3")
    )
    filename = tmp_path / "test.shp"
    write_dataframe(expected, filename, use_arrow=use_arrow)

    assert filename.exists()

    df = read_dataframe(filename)

    crs = df.crs.to_dict()
    assert crs["lat_1"] == 49.5
    assert crs["lat_2"] == 51.5
    assert crs["lon_0"] == 4.3
    assert df.crs.equals(expected.crs)


@pytest.mark.parametrize("ext", [".gpkg.zip", ".shp.zip", ".shz"])
@pytest.mark.requires_arrow_write_api
def test_write_read_zipped_ext(tmp_path, naturalearth_lowres, ext, use_arrow):
    """Run a basic read and write test on some extra (zipped) extensions."""
    if ext == ".gpkg.zip" and not GDAL_GE_37:
        pytest.skip(".gpkg.zip support requires GDAL >= 3.7")

    input_gdf = read_dataframe(naturalearth_lowres)
    output_path = tmp_path / f"test{ext}"

    write_dataframe(input_gdf, output_path, use_arrow=use_arrow)

    assert output_path.exists()
    result_gdf = read_dataframe(output_path)

    geometry_types = result_gdf.geometry.type.unique()
    if DRIVERS[ext] in DRIVERS_NO_MIXED_SINGLE_MULTI:
        assert list(geometry_types) == ["MultiPolygon"]
    else:
        assert set(geometry_types) == {"MultiPolygon", "Polygon"}

    assert_geodataframe_equal(result_gdf, input_gdf, check_index_type=False)


def test_write_read_mixed_column_values(tmp_path):
    # use_arrow=True is tested separately below
    mixed_values = ["test", 1.0, 1, datetime.now(), None, np.nan]
    geoms = [shapely.Point(0, 0) for _ in mixed_values]
    test_gdf = gp.GeoDataFrame(
        {"geometry": geoms, "mixed": mixed_values}, crs="epsg:31370"
    )
    output_path = tmp_path / "test_write_mixed_column.gpkg"
    write_dataframe(test_gdf, output_path)
    output_gdf = read_dataframe(output_path)
    assert len(test_gdf) == len(output_gdf)
    # mixed values as object dtype are currently written as strings
    # (but preserving nulls)
    expected = pd.Series(
        [str(value) if value not in (None, np.nan) else None for value in mixed_values],
        name="mixed",
    )
    assert_series_equal(output_gdf["mixed"], expected)


@requires_arrow_write_api
def test_write_read_mixed_column_values_arrow(tmp_path):
    # Arrow cannot represent a column of mixed types
    mixed_values = ["test", 1.0, 1, datetime.now(), None, np.nan]
    geoms = [shapely.Point(0, 0) for _ in mixed_values]
    test_gdf = gp.GeoDataFrame(
        {"geometry": geoms, "mixed": mixed_values}, crs="epsg:31370"
    )
    output_path = tmp_path / "test_write_mixed_column.gpkg"
    with pytest.raises(TypeError, match=".*Conversion failed for column"):
        write_dataframe(test_gdf, output_path, use_arrow=True)


@pytest.mark.requires_arrow_write_api
def test_write_read_null(tmp_path, use_arrow):
    output_path = tmp_path / "test_write_nan.gpkg"
    geom = shapely.Point(0, 0)
    test_data = {
        "geometry": [geom, geom, geom],
        "float64": [1.0, None, np.nan],
        "object_str": ["test", None, np.nan],
    }
    test_gdf = gp.GeoDataFrame(test_data, crs="epsg:31370")
    write_dataframe(test_gdf, output_path, use_arrow=use_arrow)
    result_gdf = read_dataframe(output_path)
    assert len(test_gdf) == len(result_gdf)
    assert result_gdf["float64"][0] == 1.0
    assert pd.isna(result_gdf["float64"][1])
    assert pd.isna(result_gdf["float64"][2])
    assert result_gdf["object_str"][0] == "test"
    assert pd.isna(result_gdf["object_str"][1])
    assert pd.isna(result_gdf["object_str"][2])


@pytest.mark.requires_arrow_write_api
def test_write_read_vsimem(naturalearth_lowres_vsi, use_arrow):
    path, _ = naturalearth_lowres_vsi
    mem_path = f"/vsimem/{path.name}"

    input = read_dataframe(path, use_arrow=use_arrow)
    assert len(input) == 177

    try:
        write_dataframe(input, mem_path, use_arrow=use_arrow)
        result = read_dataframe(mem_path, use_arrow=use_arrow)
        assert len(result) == 177
    finally:
        vsi_unlink(mem_path)


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
            "MultiPolygon Z (((0 0 0, 0 1 0, 1 1 0, 0 0 0)), ((1 1 1, 1 2 1, 2 2 1, 1 1 1)))",  # noqa: E501
            ["2.5D MultiPolygon", "MultiPolygon Z"],
        ),
        (
            "GeometryCollection Z (Point Z (0 0 0))",
            ["2.5D GeometryCollection", "GeometryCollection Z"],
        ),
    ],
)
@pytest.mark.requires_arrow_write_api
def test_write_geometry_z_types(tmp_path, wkt, geom_types, use_arrow):
    filename = tmp_path / "test.fgb"
    gdf = gp.GeoDataFrame(geometry=from_wkt([wkt]), crs="EPSG:4326")
    for geom_type in geom_types:
        write_dataframe(gdf, filename, use_arrow=use_arrow, geometry_type=geom_type)
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
@pytest.mark.requires_arrow_write_api
def test_write_geometry_z_types_auto(
    tmp_path, ext, test_descr, exp_geometry_type, mixed_dimensions, wkt, use_arrow
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
            write_dataframe(gdf, filename, use_arrow=use_arrow)
        return
    else:
        write_dataframe(gdf, filename, use_arrow=use_arrow)

    info = read_info(filename)
    assert info["geometry_type"] == exp_geometry_type

    result_gdf = read_dataframe(filename)
    if ext == ".geojsonl":
        result_gdf.crs = "EPSG:4326"

    assert_geodataframe_equal(gdf, result_gdf)


@pytest.mark.parametrize(
    "on_invalid, message, expected_wkt",
    [
        (
            "warn",
            "Invalid WKB: geometry is returned as None. IllegalArgumentException: "
            "Points of LinearRing do not form a closed linestring",
            None,
        ),
        ("raise", "Points of LinearRing do not form a closed linestring", None),
        ("ignore", None, None),
        ("fix", None, "POLYGON ((0 0, 0 1, 0 0))"),
    ],
)
@pytest.mark.filterwarnings("ignore:Non closed ring detected:RuntimeWarning")
def test_read_invalid_poly_ring(tmp_path, use_arrow, on_invalid, message, expected_wkt):
    if on_invalid == "fix" and not SHAPELY_GE_21:
        pytest.skip("on_invalid=fix not available for Shapely < 2.1")

    if on_invalid == "raise":
        handler = pytest.raises(shapely.errors.GEOSException, match=message)
    elif on_invalid == "warn":
        handler = pytest.warns(match=message)
    elif on_invalid in ("fix", "ignore"):
        handler = contextlib.nullcontext()
    else:
        raise ValueError(f"unknown value for on_invalid: {on_invalid}")

    # create a GeoJSON file with an invalid exterior ring
    invalid_geojson = """{
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ [ [0, 0], [0, 1] ] ]
                }
            }
        ]
    }"""

    filename = tmp_path / "test.geojson"
    with open(filename, "w") as f:
        _ = f.write(invalid_geojson)

    with handler:
        df = read_dataframe(
            filename,
            use_arrow=use_arrow,
            on_invalid=on_invalid,
        )
        if expected_wkt is None:
            assert df.geometry.iloc[0] is None
        else:
            assert df.geometry.iloc[0].wkt == expected_wkt


def test_read_multisurface(multisurface_file, use_arrow):
    if use_arrow:
        # TODO: revisit once https://github.com/geopandas/pyogrio/issues/478
        # is resolved.
        pytest.skip("Shapely + GEOS 3.13 crashes in from_wkb for this case")

        with pytest.raises(shapely.errors.GEOSException):
            # TODO(Arrow)
            # shapely fails parsing the WKB
            read_dataframe(multisurface_file, use_arrow=True)
    else:
        df = read_dataframe(multisurface_file)

        # MultiSurface should be converted to MultiPolygon
        assert df.geometry.type.tolist() == ["MultiPolygon"]


def test_read_dataset_kwargs(nested_geojson_file, use_arrow):
    # by default, nested data are not flattened
    df = read_dataframe(nested_geojson_file, use_arrow=use_arrow)

    expected = gp.GeoDataFrame(
        {
            "top_level": ["A"],
            "intermediate_level": [{"bottom_level": "B"}],
        },
        geometry=[shapely.Point(0, 0)],
        crs="EPSG:4326",
    )
    if GDAL_GE_311 and use_arrow:
        # GDAL 3.11 started to use json extension type, which is not yet handled
        # correctly in the arrow->pandas conversion (using object instead of str dtype)
        expected["intermediate_level"] = expected["intermediate_level"].astype(object)

    assert_geodataframe_equal(df, expected)

    df = read_dataframe(
        nested_geojson_file, use_arrow=use_arrow, FLATTEN_NESTED_ATTRIBUTES="YES"
    )

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


@pytest.mark.requires_arrow_write_api
def test_write_nullable_dtypes(tmp_path, use_arrow):
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
    write_dataframe(input_gdf, path, use_arrow=use_arrow)
    output_gdf = read_dataframe(path)
    # We read it back as default (non-nullable) numpy dtypes, so we cast
    # to those for the expected result
    expected = input_gdf.copy()
    expected["col2"] = expected["col2"].astype("float64")
    expected["col3"] = expected["col3"].astype("float32")
    expected["col4"] = expected["col4"].astype("float64")
    expected["col5"] = expected["col5"].astype("str")
    expected.loc[1, "col5"] = None  # pandas converts to pd.NA on line above
    assert_geodataframe_equal(output_gdf, expected)


@pytest.mark.parametrize(
    "metadata_type", ["dataset_metadata", "layer_metadata", "metadata"]
)
@pytest.mark.requires_arrow_write_api
def test_metadata_io(tmp_path, naturalearth_lowres, metadata_type, use_arrow):
    metadata = {"level": metadata_type}

    df = read_dataframe(naturalearth_lowres)

    filename = tmp_path / "test.gpkg"
    write_dataframe(df, filename, use_arrow=use_arrow, **{metadata_type: metadata})

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
@pytest.mark.requires_arrow_write_api
def test_invalid_metadata(
    tmp_path, naturalearth_lowres, metadata_type, metadata, use_arrow
):
    df = read_dataframe(naturalearth_lowres)
    with pytest.raises(ValueError, match="must be a string"):
        write_dataframe(
            df, tmp_path / "test.gpkg", use_arrow=use_arrow, **{metadata_type: metadata}
        )


@pytest.mark.parametrize("metadata_type", ["dataset_metadata", "layer_metadata"])
@pytest.mark.requires_arrow_write_api
def test_metadata_unsupported(tmp_path, naturalearth_lowres, metadata_type, use_arrow):
    """metadata is silently ignored"""

    filename = tmp_path / "test.geojson"
    write_dataframe(
        read_dataframe(naturalearth_lowres),
        filename,
        use_arrow=use_arrow,
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


@requires_pyarrow_api
@pytest.mark.skipif(
    __gdal_version__ < (3, 8, 3), reason="Arrow bool value bug fixed in GDAL >= 3.8.3"
)
@pytest.mark.parametrize("ext", ALL_EXTS)
def test_arrow_bool_roundtrip(tmp_path, ext):
    filename = tmp_path / f"test{ext}"

    kwargs = {}

    if ext == ".fgb":
        # For .fgb, spatial_index=False to avoid the rows being reordered
        kwargs["spatial_index"] = False

    df = gp.GeoDataFrame(
        {"bool_col": [True, False, True, False, True], "geometry": [Point(0, 0)] * 5},
        crs="EPSG:4326",
    )

    write_dataframe(df, filename, **kwargs)
    result = read_dataframe(filename, use_arrow=True)
    # Shapefiles do not support bool columns; these are returned as int32
    assert_geodataframe_equal(result, df, check_dtype=ext != ".shp")


@requires_pyarrow_api
@pytest.mark.skipif(
    __gdal_version__ >= (3, 8, 3), reason="Arrow bool value bug fixed in GDAL >= 3.8.3"
)
@pytest.mark.parametrize("ext", ALL_EXTS)
def test_arrow_bool_exception(tmp_path, ext):
    filename = tmp_path / f"test{ext}"

    df = gp.GeoDataFrame(
        {"bool_col": [True, False, True, False, True], "geometry": [Point(0, 0)] * 5},
        crs="EPSG:4326",
    )

    write_dataframe(df, filename)

    if ext in {".fgb", ".gpkg"}:
        # only raise exception for GPKG / FGB
        with pytest.raises(
            RuntimeError,
            match="GDAL < 3.8.3 does not correctly read boolean data values using "
            "the Arrow API",
        ):
            read_dataframe(filename, use_arrow=True)

        # do not raise exception if no bool columns are read
        read_dataframe(filename, use_arrow=True, columns=[])

    else:
        _ = read_dataframe(filename, use_arrow=True)


@requires_pyarrow_api
def test_arrow_enable_with_environment_variable(tmp_path):
    """Test if arrow can be enabled via an environment variable."""
    # Latin 1 / Western European
    encoding = "CP1252"
    text = "ÿ"
    test_path = tmp_path / "test.gpkg"

    df = gp.GeoDataFrame({text: [text], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    write_dataframe(df, test_path, encoding=encoding)

    # Without arrow, specifying the encoding is supported
    result = read_dataframe(test_path, encoding="cp1252")
    assert result is not None

    # With arrow enabled, specifying the encoding is not supported
    with use_arrow_context():
        with pytest.raises(
            ValueError, match="non-UTF-8 encoding is not supported for Arrow"
        ):
            _ = read_dataframe(test_path, encoding="cp1252")


@pytest.mark.filterwarnings("ignore:File /vsimem:RuntimeWarning")
@pytest.mark.parametrize("driver", ["GeoJSON", "GPKG"])
def test_write_memory(naturalearth_lowres, driver):
    df = read_dataframe(naturalearth_lowres)

    buffer = BytesIO()
    write_dataframe(df, buffer, driver=driver, layer="test")

    assert len(buffer.getbuffer()) > 0

    actual = read_dataframe(buffer)
    assert len(actual) == len(df)

    is_json = driver == "GeoJSON"

    assert_geodataframe_equal(
        actual,
        df,
        check_less_precise=is_json,
        check_index_type=False,
        check_dtype=not is_json,
    )

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


def test_write_memory_driver_required(naturalearth_lowres):
    df = read_dataframe(naturalearth_lowres)

    buffer = BytesIO()

    with pytest.raises(
        ValueError,
        match="driver must be provided to write to in-memory file",
    ):
        write_dataframe(df.head(1), buffer, driver=None, layer="test")

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


@pytest.mark.parametrize("driver", ["ESRI Shapefile", "OpenFileGDB"])
def test_write_memory_unsupported_driver(naturalearth_lowres, driver):
    df = read_dataframe(naturalearth_lowres)

    buffer = BytesIO()

    with pytest.raises(
        ValueError, match=f"writing to in-memory file is not supported for {driver}"
    ):
        write_dataframe(df, buffer, driver=driver, layer="test")

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


@pytest.mark.parametrize("driver", ["GeoJSON", "GPKG"])
def test_write_memory_append_unsupported(naturalearth_lowres, driver):
    df = read_dataframe(naturalearth_lowres)

    buffer = BytesIO()

    with pytest.raises(
        NotImplementedError, match="append is not supported for in-memory files"
    ):
        write_dataframe(df.head(1), buffer, driver=driver, layer="test", append=True)

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


def test_write_memory_existing_unsupported(naturalearth_lowres):
    df = read_dataframe(naturalearth_lowres)

    buffer = BytesIO(b"0000")
    with pytest.raises(
        NotImplementedError,
        match="writing to existing in-memory object is not supported",
    ):
        write_dataframe(df.head(1), buffer, driver="GeoJSON", layer="test")

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


def test_write_open_file_handle(tmp_path, naturalearth_lowres):
    """Verify that writing to an open file handle is not currently supported"""

    df = read_dataframe(naturalearth_lowres)

    # verify it fails for regular file handle
    with pytest.raises(
        NotImplementedError, match="writing to an open file handle is not yet supported"
    ):
        with open(tmp_path / "test.geojson", "wb") as f:
            write_dataframe(df.head(1), f)

    # verify it fails for ZipFile
    with pytest.raises(
        NotImplementedError, match="writing to an open file handle is not yet supported"
    ):
        with ZipFile(tmp_path / "test.geojson.zip", "w") as z:
            with z.open("test.geojson", "w") as f:
                write_dataframe(df.head(1), f)

    # Check temp file was cleaned up. Filter, as gdal keeps cache files in /vsimem/.
    assert vsi_listtree("/vsimem/", pattern="pyogrio_*") == []


@pytest.mark.parametrize("ext", ["gpkg", "geojson"])
def test_non_utf8_encoding_io(tmp_path, ext, encoded_text):
    """Verify that we write non-UTF data to the data source

    IMPORTANT: this may not be valid for the data source and will likely render
    them unusable in other tools, but should successfully roundtrip unless we
    disable writing using other encodings.

    NOTE: FlatGeobuff driver cannot handle non-UTF data in GDAL >= 3.9

    NOTE: pyarrow cannot handle non-UTF-8 characters in this way
    """

    encoding, text = encoded_text
    output_path = tmp_path / f"test.{ext}"

    df = gp.GeoDataFrame({text: [text], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    write_dataframe(df, output_path, encoding=encoding)

    # cannot open these files without specifying encoding
    with pytest.raises(UnicodeDecodeError):
        read_dataframe(output_path)

    # must provide encoding to read these properly
    actual = read_dataframe(output_path, encoding=encoding)
    assert actual.columns[0] == text
    assert actual[text].values[0] == text


@requires_pyarrow_api
@pytest.mark.parametrize("ext", ["gpkg", "geojson"])
def test_non_utf8_encoding_io_arrow_exception(tmp_path, ext, encoded_text):
    encoding, text = encoded_text
    output_path = tmp_path / f"test.{ext}"

    df = gp.GeoDataFrame({text: [text], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    write_dataframe(df, output_path, encoding=encoding)

    # cannot open these files without specifying encoding
    with pytest.raises(UnicodeDecodeError):
        read_dataframe(output_path)

    with pytest.raises(
        ValueError, match="non-UTF-8 encoding is not supported for Arrow"
    ):
        read_dataframe(output_path, encoding=encoding, use_arrow=True)


def test_non_utf8_encoding_io_shapefile(tmp_path, encoded_text, use_arrow):
    encoding, text = encoded_text

    output_path = tmp_path / "test.shp"

    df = gp.GeoDataFrame({text: [text], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    write_dataframe(df, output_path, encoding=encoding)

    # NOTE: GDAL automatically creates a cpg file with the encoding name, which
    # means that if we read this without specifying the encoding it uses the
    # correct one
    actual = read_dataframe(output_path, use_arrow=use_arrow)
    assert actual.columns[0] == text
    assert actual[text].values[0] == text

    # verify that if cpg file is not present, that user-provided encoding must be used
    output_path.with_suffix(".cpg").unlink()

    # We will assume ISO-8859-1, which is wrong
    miscoded = text.encode(encoding).decode("ISO-8859-1")

    if use_arrow:
        # pyarrow cannot decode column name with incorrect encoding
        with pytest.raises(
            DataSourceError,
            match="The file being read is not encoded in UTF-8; please use_arrow=False",
        ):
            read_dataframe(output_path, use_arrow=True)
    else:
        bad = read_dataframe(output_path, use_arrow=False)
        assert bad.columns[0] == miscoded
        assert bad[miscoded].values[0] == miscoded

    # If encoding is provided, that should yield correct text
    actual = read_dataframe(output_path, encoding=encoding, use_arrow=use_arrow)
    assert actual.columns[0] == text
    assert actual[text].values[0] == text

    # if ENCODING open option, that should yield correct text
    actual = read_dataframe(output_path, use_arrow=use_arrow, ENCODING=encoding)
    assert actual.columns[0] == text
    assert actual[text].values[0] == text


def test_encoding_read_option_collision_shapefile(naturalearth_lowres, use_arrow):
    """Providing both encoding parameter and ENCODING open option
    (even if blank) is not allowed."""

    with pytest.raises(
        ValueError, match='cannot provide both encoding parameter and "ENCODING" option'
    ):
        read_dataframe(
            naturalearth_lowres, encoding="CP936", ENCODING="", use_arrow=use_arrow
        )


def test_encoding_write_layer_option_collision_shapefile(tmp_path, encoded_text):
    """Providing both encoding parameter and ENCODING layer creation option
    (even if blank) is not allowed."""
    encoding, text = encoded_text

    output_path = tmp_path / "test.shp"
    df = gp.GeoDataFrame({text: [text], "geometry": [Point(0, 0)]}, crs="EPSG:4326")

    with pytest.raises(
        ValueError,
        match=(
            'cannot provide both encoding parameter and "ENCODING" layer creation '
            "option"
        ),
    ):
        write_dataframe(
            df, output_path, encoding=encoding, layer_options={"ENCODING": ""}
        )


def test_non_utf8_encoding_shapefile_sql(tmp_path, use_arrow):
    encoding = "CP936"

    output_path = tmp_path / "test.shp"

    mandarin = "中文"
    df = gp.GeoDataFrame(
        {mandarin: mandarin, "geometry": [Point(0, 0)]}, crs="EPSG:4326"
    )
    write_dataframe(df, output_path, encoding=encoding)

    actual = read_dataframe(
        output_path,
        sql=f"select * from test where \"{mandarin}\" = '{mandarin}'",
        use_arrow=use_arrow,
    )
    assert actual.columns[0] == mandarin
    assert actual[mandarin].values[0] == mandarin

    actual = read_dataframe(
        output_path,
        sql=f"select * from test where \"{mandarin}\" = '{mandarin}'",
        encoding=encoding,
        use_arrow=use_arrow,
    )
    assert actual.columns[0] == mandarin
    assert actual[mandarin].values[0] == mandarin


@pytest.mark.requires_arrow_write_api
def test_write_kml_file_coordinate_order(tmp_path, use_arrow):
    # confirm KML coordinates are written in lon, lat order even if CRS axis
    # specifies otherwise
    points = [Point(10, 20), Point(30, 40), Point(50, 60)]
    gdf = gp.GeoDataFrame(geometry=points, crs="EPSG:4326")
    output_path = tmp_path / "test.kml"
    write_dataframe(
        gdf, output_path, layer="tmp_layer", driver="KML", use_arrow=use_arrow
    )

    gdf_in = read_dataframe(output_path, use_arrow=use_arrow)

    assert np.array_equal(gdf_in.geometry.values, points)


@pytest.mark.requires_arrow_write_api
@pytest.mark.skipif(
    "LIBKML" not in list_drivers(),
    reason="LIBKML driver is not available and is needed to append to .kml",
)
def test_write_kml_append(tmp_path, use_arrow):
    """Append features to an existing KML file.

    Appending is only supported by the LIBKML driver, and the driver isn't
    included in the GDAL ubuntu-small images, so skip if not available.
    """
    points = [Point(10, 20), Point(30, 40), Point(50, 60)]
    gdf = gp.GeoDataFrame(geometry=points, crs="EPSG:4326")
    output_path = tmp_path / "test.kml"
    write_dataframe(
        gdf, output_path, layer="tmp_layer", driver="KML", use_arrow=use_arrow
    )

    # test appending to the existing file only if LIBKML is available
    # as it appears to fall back on LIBKML driver when appending.
    points_append = [Point(7, 8), Point(9, 10), Point(11, 12)]
    gdf_append = gp.GeoDataFrame(geometry=points_append, crs="EPSG:4326")

    write_dataframe(
        gdf_append,
        output_path,
        layer="tmp_layer",
        driver="KML",
        use_arrow=use_arrow,
        append=True,
    )
    # force_2d is used to only compare the xy dimensions of the geometry, as the LIBKML
    # driver always adds the z-dimension when the kml file is over-written.
    gdf_in_appended = read_dataframe(output_path, use_arrow=use_arrow, force_2d=True)

    assert np.array_equal(gdf_in_appended.geometry.values, points + points_append)


@pytest.mark.requires_arrow_write_api
def test_write_geojson_rfc7946_coordinates(tmp_path, use_arrow):
    points = [Point(10, 20), Point(30, 40), Point(50, 60)]
    gdf = gp.GeoDataFrame(geometry=points, crs="EPSG:4326")
    output_path = tmp_path / "test.geojson"
    write_dataframe(
        gdf,
        output_path,
        layer="tmp_layer",
        driver="GeoJSON",
        RFC7946=True,
        use_arrow=use_arrow,
    )

    gdf_in = read_dataframe(output_path, use_arrow=use_arrow)

    assert np.array_equal(gdf_in.geometry.values, points)

    # test appending to the existing file

    points_append = [Point(70, 80), Point(90, 100), Point(110, 120)]
    gdf_append = gp.GeoDataFrame(geometry=points_append, crs="EPSG:4326")

    write_dataframe(
        gdf_append,
        output_path,
        layer="tmp_layer",
        driver="GeoJSON",
        RFC7946=True,
        use_arrow=use_arrow,
        append=True,
    )

    gdf_in_appended = read_dataframe(output_path, use_arrow=use_arrow)
    assert np.array_equal(gdf_in_appended.geometry.values, points + points_append)


@pytest.mark.requires_arrow_write_api
@pytest.mark.skipif(
    not GDAL_HAS_PARQUET_DRIVER, reason="Parquet driver is not available"
)
def test_parquet_driver(tmp_path, use_arrow):
    """
    Simple test verifying the Parquet driver works if available
    """
    gdf = gp.GeoDataFrame(
        {"col": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
        crs="EPSG:4326",
    )
    output_path = tmp_path / "test.parquet"
    write_dataframe(gdf, output_path, use_arrow=use_arrow)
    result = read_dataframe(output_path, use_arrow=use_arrow)
    assert_geodataframe_equal(result, gdf)
