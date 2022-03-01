import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from distutils import log
from setuptools import setup
from setuptools.extension import Extension
import versioneer

# import Cython if available
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    cythonize = None


MIN_PYTHON_VERSION = (3, 6, 0)
MIN_GDAL_VERSION = (2, 4, 0)

build_ext = None


if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError("Python >= 3.6 is required")


# Get GDAL config from gdal-config command
def read_response(cmd):
    return subprocess.check_output(cmd).decode("utf").strip()


def get_gdal_paths():
    """Obtain the paths for compiling and linking with the GDAL C-API

    First the presence of the GDAL_INCLUDE_PATH and GDAL_LIBRARY_PATH environment
    variables is checked. If they are both present, these are taken.

    If one of the two paths was not present, gdal-config is called (it should be on the
    PATH variable). gdal-config provides all the paths.

    If no environment variables were specified or gdal-config was not found,
    no additional paths are provided to the extension. It is still possible
    to compile in this case using custom arguments to setup.py.
    """
    include_dir = os.environ.get("GDAL_INCLUDE_PATH")
    library_dir = os.environ.get("GDAL_LIBRARY_PATH")

    if include_dir and library_dir:
        return {
            "include_dirs": [include_dir],
            "library_dirs": [library_dir],
            "libraries": ["gdal_i" if platform.system() == "Windows" else "gdal"],
        }
    if include_dir or library_dir:
        log.warn(
            "If specifying the GDAL_INCLUDE_PATH or GDAL_LIBRARY_PATH environment "
            "variables, you need to specify both."
        )

    try:
        # Get libraries, etc from gdal-config (not available on Windows)
        flags = ["cflags", "libs", "version"]
        gdal_config = os.environ.get("GDAL_CONFIG", "gdal-config")
        config = {flag: read_response([gdal_config, f"--{flag}"]) for flag in flags}

        GDAL_VERSION = tuple(int(i) for i in config["version"].split("."))
        if not GDAL_VERSION > MIN_GDAL_VERSION:
            sys.exit("GDAL must be >= 2.4.x")

        include_dirs = [
            entry[2:] for entry in config["cflags"].split(" ")
        ]
        library_dirs = []
        libraries = []
        extra_link_args = []

        for entry in config["libs"].split(" "):
            if entry.startswith("-L"):
                library_dirs.append(entry[2:])
            elif entry.startswith("-l"):
                libraries.append(entry[2:])
            else:
                extra_link_args.append(entry)
        
        return {
            "include_dirs": include_dirs,
            "library_dirs": library_dirs,
            "libraries": libraries,
            "extra_link_args": extra_link_args,
        }

    except Exception as e:
        if platform.system() == "Windows":
            # Note: additional command-line parameters required to point to GDAL;
            # see the README.

            gdal_version_str = os.environ.get("GDAL_VERSION", "")
            if not gdal_version_str:
                # attempt to execute gdalinfo to find GDAL version
                # Note: gdalinfo must be on the PATH
                try:
                    gdalinfo_path = None
                    for path in os.getenv("PATH", "").split(os.pathsep):
                        matches = list(Path(path).glob("**/gdalinfo.exe"))
                        if matches:
                            gdalinfo_path = matches[0]
                            break

                    if gdalinfo_path:
                        raw_version = read_response([gdalinfo_path, "--version"]) or ""
                        m = re.search("\d+\.\d+\.\d+", raw_version)
                        if m:
                            gdal_version_str = m.group()

                except:
                    log.warn(
                        "Could not obtain version by executing 'gdalinfo --version'"
                    )

            if not gdal_version_str:
                print("GDAL_VERSION must be provided as an environment variable")
                sys.exit(1)

            GDAL_VERSION = tuple(int(i) for i in gdal_version_str.split("."))
            if not GDAL_VERSION > MIN_GDAL_VERSION:
                sys.exit("GDAL must be >= 2.4.x")

            log.info(
                "Building on Windows requires extra options to setup.py to locate GDAL files.  See the README."
            )
            return {}

        else:
            raise e


ext_modules = []

# setuptools clean does not cleanup Cython artifacts
if "clean" in sys.argv:
    if os.path.exists("build"):
        shutil.rmtree("build")

    root = Path(".")
    for ext in ["*.so", "*.pyc", "*.c", "*.cpp"]:
        for entry in root.rglob(ext):
            entry.unlink()

else:
    if cythonize is None:
        raise ImportError("Cython is required to build from source")

    ext_options = get_gdal_paths()

    ext_modules = cythonize(
        [
            Extension("pyogrio._err", ["pyogrio/_err.pyx"], **ext_options),
            Extension("pyogrio._geometry", ["pyogrio/_geometry.pyx"], **ext_options),
            Extension("pyogrio._io", ["pyogrio/_io.pyx"], **ext_options),
            Extension("pyogrio._ogr", ["pyogrio/_ogr.pyx"], **ext_options),
        ],
        compiler_directives={"language_level": "3"},
    )

    # Get numpy include directory without importing numpy at top level here
    # from: https://stackoverflow.com/a/42163080
    class build_ext(_build_ext):
        def run(self):
            try:
                import numpy

                self.include_dirs.append(numpy.get_include())
                # Call original build_ext command
                _build_ext.run(self)

            except ImportError:
                pass


version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()
cmdclass["build_ext"] = build_ext

setup(
    name="pyogrio",
    version=version,
    packages=["pyogrio"],
    url="https://github.com/brendan-ward/pyogrio",
    license="MIT",
    author="Brendan C. Ward",
    author_email="bcward@astutespruce.com",
    description="Vectorized spatial vector file format I/O using GDAL/OGR",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    python_requires=">=3.6",
    install_requires=["numpy"],
    extras_require={
        "dev": ["Cython"],
        "test": ["pytest", "pytest-cov"],
        "benchmark": ["pytest-benchmark"],
        "geopandas": ["pygeos", "geopandas"],
    },
    include_package_data=True,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
