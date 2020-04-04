"""
NOTE: this requires that all packages use the same version of GEOS.
Install each so that they use the system GEOS.
After installing geopandas, reinstall shapely via:
`pip install shapely --no-binary shapely`
"""

import geopandas as gp
import pytest

from pyogrio import read_dataframe


@pytest.mark.benchmark(group="read-pandas-modres-admin0")
def test_read_dataframe_benchmark_modres(naturalearth_modres, benchmark):
    benchmark(read_dataframe, naturalearth_modres, as_pygeos=True)


@pytest.mark.benchmark(group="read-pandas-modres-admin0")
def test_read_benchmark_geopandas_modres(naturalearth_modres, benchmark):
    benchmark(gp.read_file, naturalearth_modres)


@pytest.mark.benchmark(group="read-pandas-modres-admin1")
def test_read_dataframe_benchmark_modres1(naturalearth_modres1, benchmark):
    benchmark(read_dataframe, naturalearth_modres1, as_pygeos=True)


@pytest.mark.benchmark(group="read-pandas-modres-admin1")
def test_read_benchmark_fiona_modres1(naturalearth_modres1, benchmark):
    benchmark(gp.read_file, naturalearth_modres1)
