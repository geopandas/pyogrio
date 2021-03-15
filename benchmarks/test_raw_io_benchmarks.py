import os

import fiona
import pytest

from pyogrio.raw import read, write


def fiona_read(path, layer=None):
    """Read records from OGR data source using Fiona.

    Note: Fiona returns different information than pyogrio and we have to
    use a list here to force reading from Fiona's records generator -
    both of which incur a slight performance penalty.
    """
    with fiona.open(path, layer=layer) as src:
        list(src)


def fiona_write(path, records, **kwargs):
    with fiona.open(path, "w", **kwargs) as out:
        for record in records:
            out.write(record)


@pytest.mark.benchmark(group="read-lowres")
def test_read_lowres(naturalearth_lowres, benchmark):
    benchmark(read, naturalearth_lowres)


@pytest.mark.benchmark(group="read-lowres")
def test_read_fiona_lowres(naturalearth_lowres, benchmark):
    benchmark(fiona_read, naturalearth_lowres)


@pytest.mark.benchmark(group="read-modres-admin0")
def test_read_modres(naturalearth_modres, benchmark):
    benchmark(read, naturalearth_modres)


@pytest.mark.benchmark(group="read-modres-admin0")
def test_read_vsi_modres(naturalearth_modres_vsi, benchmark):
    benchmark(read, naturalearth_modres_vsi)


@pytest.mark.benchmark(group="read-modres-admin0")
def test_read_fiona_modres(naturalearth_modres, benchmark):
    benchmark(fiona_read, naturalearth_modres)


@pytest.mark.benchmark(group="read-modres-admin1")
def test_read_modres1(naturalearth_modres1, benchmark):
    benchmark(read, naturalearth_modres1)


@pytest.mark.benchmark(group="read-modres-admin1")
def test_read_fiona_modres1(naturalearth_modres1, benchmark):
    benchmark(fiona_read, naturalearth_modres1)


@pytest.mark.benchmark(group="read-nhd_hr")
def test_read_nhd_hr(nhd_hr, benchmark):
    benchmark(read, nhd_hr, layer="NHDFlowline")


@pytest.mark.benchmark(group="read-nhd_hr")
def test_read_fiona_nhd_hr(nhd_hr, benchmark):
    benchmark(fiona_read, nhd_hr, layer="NHDFlowline")


@pytest.mark.benchmark(group="read-subset")
def test_read_full_modres1(naturalearth_modres1, benchmark):
    benchmark(read, naturalearth_modres1)


@pytest.mark.benchmark(group="read-subset")
def test_read_no_geometry_modres1(naturalearth_modres1, benchmark):
    benchmark(read, naturalearth_modres1, read_geometry=False)


@pytest.mark.benchmark(group="read-subset")
def test_read_one_column_modres1(naturalearth_modres1, benchmark):
    benchmark(read, naturalearth_modres1, columns=["NAME"])


@pytest.mark.benchmark(group="read-subset")
def test_read_only_geometry_modres1(naturalearth_modres1, benchmark):
    benchmark(read, naturalearth_modres1, columns=[])


@pytest.mark.benchmark(group="read-subset")
def test_read_only_meta_modres1(naturalearth_modres1, benchmark):
    benchmark(read, naturalearth_modres1, columns=[], read_geometry=False)


@pytest.mark.benchmark(group="write-lowres")
def test_write_lowres_shp(tmpdir, naturalearth_lowres, benchmark):
    meta, geometry, field_data = read(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(write, filename, geometry, field_data, driver="ESRI Shapefile", **meta)


@pytest.mark.benchmark(group="write-lowres")
def test_write_lowres_gpkg(tmpdir, naturalearth_lowres, benchmark):
    meta, geometry, field_data = read(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.gpkg")
    benchmark(write, filename, geometry, field_data, driver="GPKG", **meta)


@pytest.mark.benchmark(group="write-lowres")
def test_write_lowres_geojson(tmpdir, naturalearth_lowres, benchmark):
    meta, geometry, field_data = read(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.json")
    benchmark(write, filename, geometry, field_data, driver="GeoJSON", **meta)


@pytest.mark.benchmark(group="write-lowres")
def test_write_lowres_geojsonseq(tmpdir, naturalearth_lowres, benchmark):
    meta, geometry, field_data = read(naturalearth_lowres)
    filename = os.path.join(str(tmpdir), "test.json")
    benchmark(write, filename, geometry, field_data, driver="GeoJSONSeq", **meta)


@pytest.mark.benchmark(group="write-lowres")
def test_write_fiona_lowres_shp(tmpdir, naturalearth_lowres, benchmark):
    with fiona.open(naturalearth_lowres) as source:
        crs = source.crs
        schema = source.schema
        records = list(source)

    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(
        fiona_write, filename, records, driver="ESRI Shapefile", crs=crs, schema=schema
    )


# @pytest.mark.benchmark(group="write-lowres")
# def test_write_fiona_lowres_gpkg(tmpdir, naturalearth_lowres, benchmark):
#     with fiona.open(naturalearth_lowres) as source:
#         crs = source.crs
#         schema = source.schema
#         records = list(source)

#     filename = os.path.join(str(tmpdir), "test.gpkg")
#     benchmark(fiona_write, filename, records, driver="GPKG", crs=crs, schema=schema)


# @pytest.mark.benchmark(group="write-lowres")
# def test_write_fiona_lowres_geojson(tmpdir, naturalearth_lowres, benchmark):
#     with fiona.open(naturalearth_lowres) as source:
#         crs = source.crs
#         schema = source.schema
#         records = list(source)

#     filename = os.path.join(str(tmpdir), "test.json")
#     benchmark(fiona_write, filename, records, driver="GeoJSON", crs=crs, schema=schema)


@pytest.mark.benchmark(group="write-modres")
def test_write_modres_shp(tmpdir, naturalearth_modres, benchmark):
    meta, geometry, field_data = read(naturalearth_modres)
    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(write, filename, geometry, field_data, **meta)


@pytest.mark.benchmark(group="write-modres")
def test_write_fiona_modres_shp(tmpdir, naturalearth_modres, benchmark):
    with fiona.open(naturalearth_modres) as source:
        crs = source.crs
        schema = source.schema
        records = list(source)

    filename = os.path.join(str(tmpdir), "test.shp")
    benchmark(
        fiona_write, filename, records, driver="ESRI Shapefile", crs=crs, schema=schema
    )
