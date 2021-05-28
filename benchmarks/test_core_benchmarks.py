import os

import fiona
import pytest

from pyogrio import list_layers, read_bounds, read_info


def fiona_read_info(path, layer=None):
    """Read basic info for an OGR data source using Fiona.

    NOTE: the information returned by Fiona is different, so this
    isn't entirely a fair comparison.
    """
    with fiona.open(path, layer=layer) as src:
        src.meta


@pytest.mark.benchmark(group="list-layers-single-lowres")
def test_list_layers_lowres(naturalearth_lowres, benchmark):
    benchmark(list_layers, naturalearth_lowres)


@pytest.mark.benchmark(group="list-layers-single-lowres")
def test_list_layers_fiona_lowres(naturalearth_lowres, benchmark):
    benchmark(fiona.listlayers, naturalearth_lowres)


@pytest.mark.benchmark(group="list-layers-single-modres")
def test_list_layers_modres(naturalearth_modres, benchmark):
    benchmark(list_layers, naturalearth_modres)


@pytest.mark.benchmark(group="list-layers-single-modres")
def test_list_layers_fiona_modres(naturalearth_modres, benchmark):
    benchmark(fiona.listlayers, naturalearth_modres)


@pytest.mark.benchmark(group="list-layers-nhd-hr")
def test_list_layers_multi(nhd_hr, benchmark):
    benchmark(list_layers, nhd_hr)


@pytest.mark.benchmark(group="list-layers-nhd-hr")
def test_list_layers_fiona_multi(nhd_hr, benchmark):
    benchmark(fiona.listlayers, nhd_hr)


@pytest.mark.benchmark(group="read-bounds-lowres")
def test_read_bounds_lowres(naturalearth_lowres, benchmark):
    benchmark(read_bounds, naturalearth_lowres)


@pytest.mark.benchmark(group="read-bounds-modres")
def test_read_bounds_modres(naturalearth_modres, benchmark):
    benchmark(read_bounds, naturalearth_modres)


@pytest.mark.benchmark(group="read-bounds-nhd-hr")
def test_read_bounds_nhd_hr(nhd_hr, benchmark):
    benchmark(read_bounds, nhd_hr, layer="NHDFlowline")


@pytest.mark.benchmark(group="read-info-lowres")
def test_read_info_lowres(naturalearth_lowres, benchmark):
    benchmark(read_info, naturalearth_lowres)


@pytest.mark.benchmark(group="read-info-lowres")
def test_read_info_fiona_lowres(naturalearth_lowres, benchmark):
    benchmark(fiona_read_info, naturalearth_lowres)


@pytest.mark.benchmark(group="read-info-modres")
def test_read_info_modres(naturalearth_modres, benchmark):
    benchmark(read_info, naturalearth_modres)


@pytest.mark.benchmark(group="read-info-modres")
def test_read_info_fiona_modres(naturalearth_modres, benchmark):
    benchmark(fiona_read_info, naturalearth_modres)


@pytest.mark.benchmark(group="read-info-nhd-hr")
def test_read_info_nhd_hr(nhd_hr, benchmark):
    benchmark(read_info, nhd_hr, layer="NHDFlowline")


@pytest.mark.benchmark(group="read-info-nhd-hr")
def test_fiona_read_info_nhd_hr(nhd_hr, benchmark):
    benchmark(fiona_read_info, nhd_hr, layer="NHDFlowline")

