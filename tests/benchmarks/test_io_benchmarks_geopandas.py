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
def test_read_benchmark_geopandas_lowres(naturalearth_lowres, benchmark):
    benchmark(gp.read_file, naturalearth_lowres)


@pytest.mark.benchmark(group="read-geopandas-modres-admin0")
def test_read_dataframe_benchmark_modres(naturalearth_modres, benchmark):
    benchmark(read_dataframe, naturalearth_modres)


@pytest.mark.benchmark(group="read-geopandas-modres-admin0")
def test_read_dataframe_benchmark_vsi_modres(naturalearth_modres_vsi, benchmark):
    benchmark(read_dataframe, naturalearth_modres_vsi)


@pytest.mark.benchmark(group="read-geopandas-modres-admin0")
def test_read_benchmark_geopandas_modres(naturalearth_modres, benchmark):
    benchmark(gp.read_file, naturalearth_modres)


@pytest.mark.benchmark(group="read-geopandas-modres-admin1")
def test_read_dataframe_benchmark_modres1(naturalearth_modres1, benchmark):
    benchmark(read_dataframe, naturalearth_modres1)


@pytest.mark.benchmark(group="read-geopandas-modres-admin1")
def test_read_benchmark_geopandas_modres1(naturalearth_modres1, benchmark):
    benchmark(gp.read_file, naturalearth_modres1)


@pytest.mark.benchmark(group="read-geopandas-nhd_hr")
def test_read_dataframe_benchmark_nhd_hr(nhd_hr, benchmark):
    benchmark(read_dataframe, nhd_hr, layer="NHDFlowline")


@pytest.mark.benchmark(group="read-geopandas-nhd_hr")
def test_read_benchmark_geopandas_nhd_hr(nhd_hr, benchmark):
    benchmark(gp.read_file, nhd_hr, layer="NHDFlowline")


### Write lowres Admin 0
@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_lowres_shp(tmpdir, naturalearth_lowres, benchmark):
    df = read_dataframe(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(write_dataframe, df, filename, driver="ESRI Shapefile")


@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_lowres_gpkg(tmpdir, naturalearth_lowres, benchmark):
    df = read_dataframe(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.gpkg")
    benchmark(write_dataframe, df, filename, driver="GPKG")


@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_lowres_geojson(
    tmpdir, naturalearth_lowres, benchmark
):
    df = read_dataframe(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.json")
    benchmark(write_dataframe, df, filename, driver="GeoJSON")


@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_lowres_geojsonseq(
    tmpdir, naturalearth_lowres, benchmark
):
    df = read_dataframe(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.json")
    benchmark(write_dataframe, df, filename, driver="GeoJSONSeq")


@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_geopandas_lowres_shp(
    tmpdir, naturalearth_lowres, benchmark
):
    df = gp.read_file(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(df.to_file, filename, driver="ESRI Shapefile")


@pytest.mark.benchmark(group="write-geopandas-lowres-admin0")
def test_write_dataframe_benchmark_geopandas_lowres_gpkg(
    tmpdir, naturalearth_lowres, benchmark
):
    df = gp.read_file(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(df.to_file, filename, driver="GPKG")


### Write modres Admin 0
@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_modres_shp(tmpdir, naturalearth_modres, benchmark):
    df = read_dataframe(naturalearth_modres)
    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(write_dataframe, df, filename, driver="ESRI Shapefile")


@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_modres_gpkg(tmpdir, naturalearth_modres, benchmark):
    df = read_dataframe(naturalearth_modres)
    filename = os.path.join(str(tmpdir), "test.gpkg")
    benchmark(write_dataframe, df, filename, driver="GPKG")


@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_modres_geojson(
    tmpdir, naturalearth_modres, benchmark
):
    df = read_dataframe(naturalearth_modres)
    filename = os.path.join(str(tmpdir), "test.json")
    benchmark(write_dataframe, df, filename, driver="GeoJSON")


@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_modres_geojsonseq(
    tmpdir, naturalearth_modres, benchmark
):
    df = read_dataframe(naturalearth_modres)
    filename = os.path.join(str(tmpdir), "test.json")
    benchmark(write_dataframe, df, filename, driver="GeoJSONSeq")


@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_geopandas_modres_shp(
    tmpdir, naturalearth_modres, benchmark
):
    df = gp.read_file(naturalearth_modres)
    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(df.to_file, filename, driver="ESRI Shapefile")


@pytest.mark.benchmark(group="write-geopandas-modres-admin0")
def test_write_dataframe_benchmark_geopandas_modres_gpkg(
    tmpdir, naturalearth_modres, benchmark
):
    df = gp.read_file(naturalearth_modres)
    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(df.to_file, filename, driver="GPKG")


### Write NHD
@pytest.mark.benchmark(group="write-geopandas-nhd_hr")
def test_write_dataframe_benchmark_nhd_shp(tmpdir, nhd_hr, benchmark):
    layer = "NHDFlowline"
    df = read_dataframe(nhd_hr, layer=layer)

    # Datetime not currently supported
    df = df.drop(columns="FDate")

    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(write_dataframe, df, filename, layer=layer, driver="ESRI Shapefile")


@pytest.mark.benchmark(group="write-geopandas-nhd_hr")
def test_write_dataframe_benchmark_geopandas_nhd_shp(tmpdir, nhd_hr, benchmark):
    layer = "NHDFlowline"
    df = gp.read_file(nhd_hr, layer=layer)

    # Datetime not currently supported by pyogrio, so drop here too so that the
    # benchmark is fair.
    df = df.drop(columns="FDate")

    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(df.to_file, filename, layer=layer, driver="ESRI Shapefile")
