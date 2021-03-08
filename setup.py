import os
from pathlib import Path
import shutil
import subprocess
import sys

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.extension import Extension


MIN_GDAL_VERSION = (2, 4, 0)


# Get GDAL config from gdal-config command
def read_response(cmd):
    return subprocess.check_output(cmd).decode("utf").strip()


ext_options = {
    "include_dirs": [],
    "library_dirs": [],
    "libraries": [],
    "extra_link_args": [],
}


# setuptools clean does not cleanup Cython artifacts
if "clean" in sys.argv:
    if os.path.exists("build"):
        shutil.rmtree("build")

    root = Path(".")
    for ext in ["*.so", "*.pyc", "*.c", "*.cpp"]:
        for entry in root.rglob(ext):
            entry.unlink()

else:
    try:
        # Get libraries, etc from gdal-config (not available on Windows)
        flags = ["cflags", "libs", "version"]
        gdal_config = os.environ.get("GDAL_CONFIG", "gdal-config")
        config = {flag: read_response([gdal_config, f"--{flag}"]) for flag in flags}

        GDAL_VERSION = tuple(int(i) for i in config["version"].split("."))

    except Exception as e:
        if sys.platform == "win32":
            # try to get GDAL version from command line
            if 'GDAL_VERSION' in os.environ:
                GDAL_VERSION = tuple(int(i) for i in os.environ.get('GDAL_VERSION', '').split('.'))

            else:
                raise ValueError("GDAL_VERSION must be provided as an environment variable")

        else:
            raise e

    if not GDAL_VERSION > MIN_GDAL_VERSION:
        sys.exit("GDAL must be >= 2.4.x")

    ext_options["include_dirs"] = [entry[2:] for entry in config["cflags"].split(" ")]

    for entry in config["libs"].split(" "):
        if entry.startswith("-L"):
            ext_options["library_dirs"].append(entry[2:])
        elif entry.startswith("-l"):
            ext_options["libraries"].append(entry[2:])
        else:
            ext_options["extra_link_args"].append(entry)


# Get numpy include directory without importing numpy at top level here
# from: https://stackoverflow.com/a/42163080
class CustomBuildExtCommand(build_ext):
    def run(self):
        try:
            import numpy

            self.include_dirs.append(numpy.get_include())
            # Call original build_ext command
            build_ext.run(self)

        except ImportError:
            pass


setup(
    name="pyogrio",
    version="0.1.0",
    packages=["pyogrio"],
    url="https://github.com/brendan-ward/pyogrio",
    license="MIT",
    author="Brendan C. Ward",
    author_email="bcward@astutespruce.com",
    description="Vectorized spatial vector file format I/O using GDAL/OGR",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    python_requires=">=3",
    install_requires=["numpy", "geopandas"],
    extras_require={
        "dev": ["Cython", "pygeos"],
        "test": ["pytest", "pytest-cov"],
        "benchmark": ["fiona", "pytest-benchmark"]
    },
    include_package_data=True,
    cmdclass={"build_ext": CustomBuildExtCommand},
    ext_modules=cythonize(
        [
            Extension("pyogrio._err", ["pyogrio/_err.pyx"], **ext_options),
            Extension("pyogrio._geometry", ["pyogrio/_geometry.pyx"], **ext_options),
            Extension("pyogrio._io", ["pyogrio/_io.pyx"], **ext_options),
            Extension("pyogrio._ogr", ["pyogrio/_ogr.pyx"], **ext_options),
        ],
        compiler_directives={"language_level": "3"},
    ),
)
