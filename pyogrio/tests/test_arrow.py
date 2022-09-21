import pytest

from pyogrio import __gdal_version__, read_dataframe

try:
    from geopandas.testing import assert_geodataframe_equal
except ImportError:
    pass


pytest.importorskip("geopandas")
pytest.importorskip("pyarrow")

pytestmark = pytest.mark.skipif(
    __gdal_version__ < (3, 6, 0), reason="Arrow tests require GDAL>=3.6"
)


def test_read_arrow(naturalearth_lowres_all_ext):
    result = read_dataframe(naturalearth_lowres_all_ext, use_arrow=True)
    expected = read_dataframe(naturalearth_lowres_all_ext, use_arrow=False)

    if naturalearth_lowres_all_ext.suffix == ".gpkg":
        fid_col = "fid"
    else:
        fid_col = "OGC_FID"

    assert fid_col in result.columns
    result = result.drop(columns=[fid_col])

    if naturalearth_lowres_all_ext.suffix.startswith(".geojson"):
        check_less_precise = True
    else:
        check_less_precise = False
    assert_geodataframe_equal(result, expected, check_less_precise=check_less_precise)


def test_read_arrow_columns(naturalearth_lowres):
    result = read_dataframe(naturalearth_lowres, use_arrow=True, columns=["continent"])
    assert result.columns.tolist() == ["OGC_FID", "continent", "geometry"]
