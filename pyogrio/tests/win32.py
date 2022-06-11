"""Run pytest tests manually on Windows due to import errors
"""
from pathlib import Path
import platform
from tempfile import TemporaryDirectory


data_dir = Path(__file__).parent.resolve() / "fixtures"

if platform.system() == "Windows":

    naturalearth_lowres = data_dir / Path("naturalearth_lowres/naturalearth_lowres.shp")
    test_fgdb_vsi = f"/vsizip/{data_dir}/test_fgdb.gdb.zip"

    from pyogrio.tests.test_core import test_read_info

    try:
        test_read_info(naturalearth_lowres)
    except Exception as ex:
        print(ex)

    from pyogrio.tests.test_raw_io import (
        test_read,
        test_read_no_geometry,
        test_read_columns,
        test_read_skip_features,
        test_read_max_features,
        test_read_where,
        test_read_where_invalid,
        test_write,
        test_write_gpkg,
        test_write_geojson,
    )

    try:
        test_read(naturalearth_lowres)
    except Exception as ex:
        print(ex)

    try:
        test_read_no_geometry(naturalearth_lowres)
    except Exception as ex:
        print(ex)

    try:
        test_read_columns(naturalearth_lowres)
    except Exception as ex:
        print(ex)

    try:
        test_read_skip_features(naturalearth_lowres)
    except Exception as ex:
        print(ex)

    try:
        test_read_max_features(naturalearth_lowres)
    except Exception as ex:
        print(ex)

    try:
        test_read_where(naturalearth_lowres)
    except Exception as ex:
        print(ex)

    try:
        test_read_where_invalid(naturalearth_lowres)
    except Exception as ex:
        print(ex)

    with TemporaryDirectory() as tmpdir:
        try:
            test_write(tmpdir, naturalearth_lowres)
        except Exception as ex:
            print(ex)

    with TemporaryDirectory() as tmpdir:
        try:
            test_write_gpkg(tmpdir, naturalearth_lowres)
        except Exception as ex:
            print(ex)

    with TemporaryDirectory() as tmpdir:
        try:
            test_write_geojson(tmpdir, naturalearth_lowres)
        except Exception as ex:
            print(ex)
