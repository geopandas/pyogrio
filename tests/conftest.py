from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from geopandas.datasets import get_path

import pytest


data_dir = Path(__file__).parent.resolve() / "fixtures/datasets"


@pytest.fixture(scope="session")
def naturalearth_lowres():
    return Path(get_path("naturalearth_lowres"))


@pytest.fixture(scope="function")
def naturalearth_lowres_vsi(tmp_path, naturalearth_lowres):
    """Wrap naturalearth_lowres as a zip file for vsi tests"""

    path = tmp_path / f"{naturalearth_lowres.name}.zip"
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED, compresslevel=5) as out:
        # out.write(naturalearth_lowres, naturalearth_lowres.name)
        for ext in ["dbf", "prj", "shp", "shx"]:
            filename = f"{naturalearth_lowres.stem}.{ext}"
            out.write(naturalearth_lowres.parent / filename, filename)

    return f"/vsizip/{path}/{naturalearth_lowres.name}"
