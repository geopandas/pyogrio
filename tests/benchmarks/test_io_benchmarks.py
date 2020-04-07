import fiona
import pytest

from pyogrio import read, list_layers


def fiona_read(path):
    with fiona.open(path) as src:
        list(src)


@pytest.mark.benchmark(group="list-layers-single")
def test_list_layers_benchmark_lowres(naturalearth_lowres, benchmark):
    benchmark(list_layers, naturalearth_lowres)


@pytest.mark.benchmark(group="list-layers-single")
def test_list_layers_benchmark_modres(naturalearth_modres, benchmark):
    benchmark(list_layers, naturalearth_modres)


@pytest.mark.benchmark(group="list-layers-single")
def test_list_layers_benchmark_fiona_lowres(naturalearth_lowres, benchmark):
    benchmark(fiona.listlayers, naturalearth_lowres)


@pytest.mark.benchmark(group="list-layers-single")
def test_list_layers_benchmark_fiona_modres(naturalearth_modres, benchmark):
    benchmark(fiona.listlayers, naturalearth_modres)


@pytest.mark.benchmark(group="list-layers-multi")
def test_list_layers_benchmark_multi(nhd_hr, benchmark):
    benchmark(list_layers, nhd_hr)


@pytest.mark.benchmark(group="list-layers-multi")
def test_list_layers_benchmark_fiona_multi(nhd_hr, benchmark):
    benchmark(fiona.listlayers, nhd_hr)


@pytest.mark.benchmark(group="read-lowres")
def test_read_benchmark_lowres(naturalearth_lowres, benchmark):
    benchmark(read, naturalearth_lowres)


@pytest.mark.benchmark(group="read-lowres")
def test_read_benchmark_fiona_lowres(naturalearth_lowres, benchmark):
    benchmark(fiona_read, naturalearth_lowres)


@pytest.mark.benchmark(group="read-modres-admin0")
def test_read_benchmark_modres(naturalearth_modres, benchmark):
    benchmark(read, naturalearth_modres)


@pytest.mark.benchmark(group="read-modres-admin0")
def test_read_benchmark_fiona_modres(naturalearth_modres, benchmark):
    benchmark(fiona_read, naturalearth_modres)


@pytest.mark.benchmark(group="read-modres-admin1")
def test_read_benchmark_modres1(naturalearth_modres1, benchmark):
    benchmark(read, naturalearth_modres1)


@pytest.mark.benchmark(group="read-modres-admin1")
def test_read_benchmark_fiona_modres1(naturalearth_modres1, benchmark):
    benchmark(fiona_read, naturalearth_modres1)
