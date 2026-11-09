import json
import warnings

from pyogrio import _warnings, read_dataframe, read_info
from pyogrio.raw import read

import pytest


@pytest.fixture
def warning_dataset(tmp_path):
    path = tmp_path / "point.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [1, 2]},
                        "properties": {"value": 3},
                    }
                ],
            }
        )
    )
    return path


def external_call(tmp_path, operation, path, wrapper):
    if wrapper == "internal":
        namespace = {"operation": operation}
        exec(
            compile(
                "def wrapped(path, **kwargs):\n    return operation(path, **kwargs)\n",
                _warnings.__file__,
                "exec",
            ),
            namespace,
        )
        operation = namespace["wrapped"]

    if wrapper == "external":
        source = (
            "def wrapped():\n"
            '    return operation(path, INVALID="YES")\n'
            "result = wrapped()\n"
        )
        line = 2
    else:
        source = 'result = operation(path, INVALID="YES")\n'
        line = 1
    caller = tmp_path / "external_caller.py"
    caller.write_text(source)
    namespace = {"operation": operation, "path": path, "__name__": "external_user"}
    exec(compile(source, str(caller), "exec"), namespace)
    assert namespace["result"] is not None
    return caller, line


@pytest.mark.parametrize("operation", [read_info, read, read_dataframe])
@pytest.mark.parametrize("wrapper", ["direct", "internal", "external"])
@pytest.mark.parametrize("legacy", [False, True])
def test_gdal_warning_callsite(
    tmp_path, warning_dataset, monkeypatch, operation, wrapper, legacy
):
    if operation is read_dataframe:
        pytest.importorskip("geopandas")
    if legacy:
        monkeypatch.setattr(_warnings, "_PY_GE_312", False)
    with pytest.warns(
        RuntimeWarning, match="does not support open option INVALID"
    ) as seen:
        caller, line = external_call(tmp_path, operation, warning_dataset, wrapper)
    assert len(seen) == 1
    assert seen[0].filename == str(caller)
    assert seen[0].lineno == line


@pytest.mark.parametrize("legacy", [False, True])
def test_gdal_warning_ignore_filter(tmp_path, warning_dataset, monkeypatch, legacy):
    if legacy:
        monkeypatch.setattr(_warnings, "_PY_GE_312", False)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "ignore", ".*does not support open option INVALID", RuntimeWarning
        )
        external_call(tmp_path, read_info, warning_dataset, "direct")
    assert seen == []


@pytest.mark.parametrize("legacy", [False, True])
def test_gdal_warning_default_filter(tmp_path, warning_dataset, monkeypatch, legacy):
    if legacy:
        monkeypatch.setattr(_warnings, "_PY_GE_312", False)
    source = 'for i in range(2):\n    result = operation(path, INVALID="YES")\n'
    namespace = {
        "operation": read_info,
        "path": warning_dataset,
        "__name__": "external_user",
    }
    caller = tmp_path / "external_caller.py"
    caller.write_text(source)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("default")
        exec(compile(source, str(caller), "exec"), namespace)
    assert len(seen) == 1
    assert seen[0].category is RuntimeWarning
    assert seen[0].filename == str(caller)
    assert seen[0].lineno == 2


@pytest.mark.parametrize("legacy", [False, True])
def test_package_prefix_does_not_match_neighbor(tmp_path, monkeypatch, legacy):
    if legacy:
        monkeypatch.setattr(_warnings, "_PY_GE_312", False)
    # A neighboring filename sharing the package's basename is still external.
    caller = _warnings._PACKAGE_PREFIXES[0].rstrip("/\\") + "_user.py"
    with pytest.warns(RuntimeWarning, match="ordinary warning") as seen:
        exec(
            compile('warn("ordinary warning")', caller, "exec"),
            {"warn": _warnings.warn},
        )
    assert seen[0].filename == caller
    assert seen[0].lineno == 1
