import math

import pytest

from pyogrio import __gdal_version__, read_dataframe
from pyogrio.raw import open_arrow, read_arrow

try:
    import pandas as pd
    from pandas.testing import assert_frame_equal, assert_index_equal
    from geopandas.testing import assert_geodataframe_equal
except ImportError:
    pass


pytest.importorskip("geopandas")
pyarrow = pytest.importorskip("pyarrow")

pytestmark = pytest.mark.skipif(
    __gdal_version__ < (3, 6, 0), reason="Arrow tests require GDAL>=3.6"
)


def test_read_arrow(naturalearth_lowres_all_ext):
    result = read_dataframe(naturalearth_lowres_all_ext, use_arrow=True)
    expected = read_dataframe(naturalearth_lowres_all_ext, use_arrow=False)

    if naturalearth_lowres_all_ext.suffix.startswith(".geojson"):
        check_less_precise = True
    else:
        check_less_precise = False
    assert_geodataframe_equal(result, expected, check_less_precise=check_less_precise)


@pytest.mark.parametrize("skip_features, expected", [(10, 167), (200, 0)])
def test_read_arrow_skip_features(naturalearth_lowres, skip_features, expected):
    table = read_arrow(naturalearth_lowres, skip_features=skip_features)[1]
    assert len(table) == expected


@pytest.mark.parametrize(
    "max_features, expected", [(0, 0), (10, 10), (200, 177), (100000, 177)]
)
def test_read_arrow_max_features(naturalearth_lowres, max_features, expected):
    table = read_arrow(naturalearth_lowres, max_features=max_features)[1]
    assert len(table) == expected


def test_read_arrow_fid(naturalearth_lowres_all_ext):
    kwargs = {"use_arrow": True, "where": "fid >= 2 AND fid <= 3"}

    df = read_dataframe(naturalearth_lowres_all_ext, fid_as_index=False, **kwargs)
    assert_index_equal(df.index, pd.RangeIndex(0, 2))

    df = read_dataframe(naturalearth_lowres_all_ext, fid_as_index=True, **kwargs)
    assert_index_equal(df.index, pd.Index([2, 3], name="fid"))


def test_read_arrow_columns(naturalearth_lowres):
    result = read_dataframe(naturalearth_lowres, use_arrow=True, columns=["continent"])
    assert result.columns.tolist() == ["continent", "geometry"]


def test_read_arrow_layer_without_geometry(test_fgdb_vsi):
    result = read_dataframe(test_fgdb_vsi, layer="basetable", use_arrow=True)
    assert type(result) is pd.DataFrame


def test_read_arrow_ignore_geometry(naturalearth_lowres):
    result = read_dataframe(naturalearth_lowres, use_arrow=True, read_geometry=False)
    assert type(result) is pd.DataFrame

    expected = read_dataframe(naturalearth_lowres, use_arrow=True).drop(
        columns=["geometry"]
    )
    assert_frame_equal(result, expected)


def test_read_arrow_nested_types(test_ogr_types_list):
    # with arrow, list types are supported
    result = read_dataframe(test_ogr_types_list, use_arrow=True)
    assert "list_int64" in result.columns
    assert result["list_int64"][0].tolist() == [0, 1]


def test_read_arrow_raw(naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)
    assert isinstance(meta, dict)
    assert isinstance(table, pyarrow.Table)


def test_open_arrow(naturalearth_lowres):
    with open_arrow(naturalearth_lowres) as (meta, reader):
        assert isinstance(meta, dict)
        assert isinstance(reader, pyarrow.RecordBatchReader)
        assert isinstance(reader.read_all(), pyarrow.Table)


def test_open_arrow_batch_size(naturalearth_lowres):
    meta, table = read_arrow(naturalearth_lowres)
    batch_size = math.ceil(len(table) / 2)

    with open_arrow(naturalearth_lowres, batch_size=batch_size) as (meta, reader):
        assert isinstance(meta, dict)
        assert isinstance(reader, pyarrow.RecordBatchReader)
        count = 0
        tables = []
        for table in reader:
            tables.append(table)
            count += 1

        assert count == 2, "Should be two batches given the batch_size parameter"
        assert len(tables[0]) == batch_size, "First table should match the batch size"


@pytest.mark.parametrize("skip_features, max_features", [(10, None), (0, 10), (10, 10)])
def test_open_arrow_skip_features_max_features_unsupported(
    naturalearth_lowres, skip_features, max_features
):
    """skip_features and max_features are not supported for the Arrow stream
    interface"""
    with pytest.raises(
        ValueError,
        match="specifying 'skip_features' or 'max_features' is not supported for Arrow",
    ):
        with open_arrow(
            naturalearth_lowres, skip_features=skip_features, max_features=max_features
        ) as (meta, reader):
            pass
