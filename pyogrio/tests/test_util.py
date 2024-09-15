from pathlib import Path

from pyogrio import vsi_listtree, vsi_unlink
from pyogrio.raw import read, write
from pyogrio.util import vsimem_rmtree_toplevel

import pytest


def test_vsimem_rmtree_toplevel(naturalearth_lowres):
    # Prepare test data in /vsimem/
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta["spatial_index"] = False
    meta["geometry_type"] = "MultiPolygon"
    test_dir_path = Path(f"/vsimem/test/{naturalearth_lowres.stem}.gpkg")
    test_dir2_path = Path(f"/vsimem/test2/test2/{naturalearth_lowres.stem}.gpkg")

    write(test_dir_path, geometry, field_data, **meta)
    write(test_dir2_path, geometry, field_data, **meta)

    # Check if everything was created properly with listtree
    files = vsi_listtree("/vsimem/")
    assert test_dir_path.as_posix() in files
    assert test_dir2_path.as_posix() in files

    # Test deleting parent dir of file in single directory
    vsimem_rmtree_toplevel(test_dir_path)
    files = vsi_listtree("/vsimem/")
    assert test_dir_path.parent.as_posix() not in files
    assert test_dir2_path.as_posix() in files

    # Test deleting top-level dir of file in a subdirectory
    vsimem_rmtree_toplevel(test_dir2_path)
    assert test_dir2_path.as_posix() not in vsi_listtree("/vsimem/")


def test_vsimem_rmtree_toplevel_error(naturalearth_lowres):
    # Prepare test data in /vsimem
    meta, _, geometry, field_data = read(naturalearth_lowres)
    meta["spatial_index"] = False
    meta["geometry_type"] = "MultiPolygon"
    test_file_path = Path(f"/vsimem/pyogrio_test_{naturalearth_lowres.stem}.gpkg")

    write(test_file_path, geometry, field_data, **meta)
    assert test_file_path.as_posix() in vsi_listtree("/vsimem/")

    # Deleting parent dir of non-existent file should raise an error.
    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        vsimem_rmtree_toplevel("/vsimem/test/non-existent.gpkg")

    # File should still be there
    assert test_file_path.as_posix() in vsi_listtree("/vsimem/")

    # Cleanup.
    vsi_unlink(test_file_path)
    assert test_file_path not in vsi_listtree("/vsimem/")
