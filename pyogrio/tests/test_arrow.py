import pytest

from pyogrio import __gdal_version__, read_dataframe
from pyogrio.raw import read_arrow

try:
    import pandas as pd
    from pandas.testing import assert_frame_equal
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


def test_read_arrow_fid(naturalearth_lowres_all_ext):
    result = read_dataframe(
        naturalearth_lowres_all_ext, use_arrow=True, fid_as_index=True
    )

    if naturalearth_lowres_all_ext.suffix == ".gpkg":
        fid_col = "fid"
    else:
        fid_col = "OGC_FID"

    assert fid_col in result.columns


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
