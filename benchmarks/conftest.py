from pathlib import Path

import pytest


data_dir = Path(__file__).parent.resolve() / "fixtures"


@pytest.fixture(scope="session")
def naturalearth_lowres():
    return data_dir / "ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"


@pytest.fixture(scope="session")
def naturalearth_modres():
    return data_dir / "ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"


@pytest.fixture(scope="session")
def naturalearth_modres_vsi():
    path = data_dir / "ne_10m_admin_0_countries.zip/ne_10m_admin_0_countries.shp"
    return f"/vsizip/{path}"


@pytest.fixture(scope="session")
def naturalearth_lowres1():
    return (
        data_dir
        / "ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp"
    )


@pytest.fixture(scope="session")
def naturalearth_modres1():
    return (
        data_dir / "ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"
    )


@pytest.fixture(scope="session")
def nhd_wbd():
    return data_dir / "WBD_17_HU2_GDB/WBD_17_HU2_GDB.gdb"


@pytest.fixture(scope="session")
def nhd_hr():
    return data_dir / "NHDPLUS_H_1704_HU4_GDB/NHDPLUS_H_1704_HU4_GDB.gdb"
