"""
NOTE: this requires that all packages use the same version of GEOS.
Install each so that they use the system GEOS.
After installing geopandas, reinstall shapely via:
`pip install shapely --no-binary shapely`
"""

import os

import geopandas as gp
import pytest

from pyogrio.geopandas import read_dataframe, write_dataframe


@pytest.mark.benchmark(group="read-geopandas-lowres-admin0")
def test_read_dataframe_benchmark_lowres(naturalearth_lowres, benchmark):
    benchmark(read_dataframe, naturalearth_lowres)


@pytest.mark.benchmark(group="read-geopandas-lowres-admin0")
def test_read_dataframe_benchmark_geopandas_lowres(naturalearth_lowres, benchmark):
    benchmark(gp.read_file, naturalearth_lowres)


@pytest.mark.benchmark(group="read-geopandas-modres-admin0")
def test_read_dataframe_benchmark_modres(naturalearth_modres, benchmark):
    benchmark(read_dataframe, naturalearth_modres)


@pytest.mark.benchmark(group="read-geopandas-modres-admin0")
def test_read_dataframe_benchmark_vsi_modres(naturalearth_modres_vsi, benchmark):
    benchmark(read_dataframe, naturalearth_modres_vsi)


@pytest.mark.benchmark(group="read-geopandas-modres-admin0")
def test_read_dataframe_benchmark_geopandas_modres(naturalearth_modres, benchmark):
    benchmark(gp.read_file, naturalearth_modres)


@pytest.mark.benchmark(group="read-geopandas-modres-admin1")
def test_read_dataframe_benchmark_modres1(naturalearth_modres1, benchmark):
    benchmark(read_dataframe, naturalearth_modres1)


@pytest.mark.benchmark(group="read-geopandas-modres-admin1")
def test_read_dataframe_benchmark_geopandas_modres1(naturalearth_modres1, benchmark):
    benchmark(gp.read_file, naturalearth_modres1)


@pytest.mark.benchmark(group="read-geopandas-nhd_hr")
def test_read_dataframe_benchmark_nhd_hr(nhd_hr, benchmark):
    benchmark(read_dataframe, nhd_hr, layer="NHDFlowline")


@pytest.mark.benchmark(group="read-geopandas-nhd_hr")
def test_read_dataframe_benchmark_geopandas_nhd_hr(nhd_hr, benchmark):
    benchmark(gp.read_file, nhd_hr, layer="NHDFlowline")


### Write lowres Admin 0
@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_lowres_shp(tmp_path, naturalearth_lowres, benchmark):
    df = read_dataframe(naturalearth_lowres)
    benchmark(write_dataframe, df, tmp_path / "test.shp", driver="ESRI Shapefile")


@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_lowres_gpkg(tmp_path, naturalearth_lowres, benchmark):
    df = read_dataframe(naturalearth_lowres)
    benchmark(write_dataframe, df, tmp_path / "test.gpkg", driver="GPKG")


@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_lowres_geojson(
    tmp_path, naturalearth_lowres, benchmark
):
    df = read_dataframe(naturalearth_lowres)
    benchmark(write_dataframe, df, tmp_path / "test.json", driver="GeoJSON")


@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_lowres_geojsonseq(
    tmp_path, naturalearth_lowres, benchmark
):
    df = read_dataframe(naturalearth_lowres)
    benchmark(write_dataframe, df, tmp_path / "test.json", driver="GeoJSONSeq")


@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_geopandas_lowres_shp(
    tmp_path, naturalearth_lowres, benchmark
):
    df = gp.read_file(naturalearth_lowres)
    benchmark(df.to_file, tmp_path / "test.shp", driver="ESRI Shapefile")


@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_geopandas_lowres_gpkg(
    tmp_path, naturalearth_lowres, benchmark
):
    df = gp.read_file(naturalearth_lowres)
    benchmark(df.to_file, tmp_path / "test.gpkg", driver="GPKG")


### Write modres Admin 0
@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_modres_shp(tmp_path, naturalearth_modres, benchmark):
    df = read_dataframe(naturalearth_modres)
    benchmark(write_dataframe, df, tmp_path / "test.shp", driver="ESRI Shapefile")


@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_modres_gpkg(tmp_path, naturalearth_modres, benchmark):
    df = read_dataframe(naturalearth_modres)
    benchmark(write_dataframe, df, tmp_path / "test.gpkg", driver="GPKG")


@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_modres_geojson(
    tmp_path, naturalearth_modres, benchmark
):
    df = read_dataframe(naturalearth_modres)
    benchmark(write_dataframe, df, tmp_path / "test.json", driver="GeoJSON")


@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_modres_geojsonseq(
    tmp_path, naturalearth_modres, benchmark
):
    df = read_dataframe(naturalearth_modres)
    benchmark(write_dataframe, df, tmp_path / "test.json", driver="GeoJSONSeq")


@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_geopandas_modres_shp(
    tmp_path, naturalearth_modres, benchmark
):
    df = gp.read_file(naturalearth_modres)
    benchmark(df.to_file, tmp_path / "test.shp", driver="ESRI Shapefile")


@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_geopandas_modres_gpkg(
    tmp_path, naturalearth_modres, benchmark
):
    df = gp.read_file(naturalearth_modres)
    benchmark(df.to_file, tmp_path / "test.gpkg", driver="GPKG")


### Write NHD
@pytest.mark.filterwarnings("ignore: RuntimeWarning")
@pytest.mark.benchmark(group="write-geopandas-nhd_hr")
def test_write_dataframe_benchmark_nhd_shp(tmp_path, nhd_hr, benchmark):
    layer = "NHDFlowline"
    df = read_dataframe(nhd_hr, layer=layer)

    # Datetime not currently supported
    df = df.drop(columns="FDate")

    benchmark(write_dataframe, df, tmp_path / "test.shp", layer=layer, driver="ESRI Shapefile")


@pytest.mark.filterwarnings("ignore: RuntimeWarning")
@pytest.mark.benchmark(group="write-geopandas-nhd_hr")
def test_write_dataframe_benchmark_geopandas_nhd_shp(tmp_path, nhd_hr, benchmark):
    layer = "NHDFlowline"
    df = gp.read_file(nhd_hr, layer=layer)

    # Datetime not currently supported by pyogrio, so drop here too so that the
    # benchmark is fair.
    df = df.drop(columns="FDate")

    benchmark(df.to_file, tmp_path/"test.shp", layer=layer, driver="ESRI Shapefile")
