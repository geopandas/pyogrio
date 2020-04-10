import fiona
import pytest

from pyogrio import read, list_layers


def fiona_read(path, layer=None):
    with fiona.open(path, layer=layer) as src:
        list(src)


@pytest.mark.benchmark(group="list-layers-single")
def test_list_layers_lowres(naturalearth_lowres, benchmark):
    benchmark(list_layers, naturalearth_lowres)


@pytest.mark.benchmark(group="list-layers-single")
def test_list_layers_modres(naturalearth_modres, benchmark):
    benchmark(list_layers, naturalearth_modres)


@pytest.mark.benchmark(group="list-layers-single")
def test_list_layers_fiona_lowres(naturalearth_lowres, benchmark):
    benchmark(fiona.listlayers, naturalearth_lowres)


@pytest.mark.benchmark(group="list-layers-single")
def test_list_layers_fiona_modres(naturalearth_modres, benchmark):
    benchmark(fiona.listlayers, naturalearth_modres)


@pytest.mark.benchmark(group="list-layers-multi")
def test_list_layers_multi(nhd_hr, benchmark):
    benchmark(list_layers, nhd_hr)


@pytest.mark.benchmark(group="list-layers-multi")
def test_list_layers_fiona_multi(nhd_hr, benchmark):
    benchmark(fiona.listlayers, nhd_hr)


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
