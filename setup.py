import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from distutils import log
from setuptools import setup, find_packages
from setuptools.extension import Extension
import versioneer

# import Cython if available
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    cythonize = None


MIN_PYTHON_VERSION = (3, 8, 0)
MIN_GDAL_VERSION = (2, 4, 0)

build_ext = None


if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError("Python >= 3.8 is required")


def copy_data_tree(datadir, destdir):
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    shutil.copytree(datadir, destdir)


# Get GDAL config from gdal-config command
def read_response(cmd):
    return subprocess.check_output(cmd).decode("utf").strip()


def get_gdal_paths():
    """Obtain the paths for compiling and linking with the GDAL C-API

    GDAL_INCLUDE_PATH and GDAL_LIBRARY_PATH environment variables are used
    if both are present.

    If those variables are not present, gdal-config is called (it should be
    on the PATH variable). gdal-config provides all the paths.

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

        gdal_version = tuple(int(i) for i in config["version"].strip("dev").split("."))
        if not gdal_version >= MIN_GDAL_VERSION:
            sys.exit("GDAL must be >= 2.4.x")

        include_dirs = [entry[2:] for entry in config["cflags"].split(" ")]
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
            log.info(
                "Building on Windows requires extra options to setup.py to locate GDAL files.  See the README."
            )
            return {}

        else:
            raise e


def get_gdal_version():
    """
    Obtain the GDAL version.

    On Linux/MacOS it will first try 'gdal-config --version'. Next, for
    all platforms it will check the GDAL_VERSION environment variable.
    On Windows it will still try 'gdalinfo --version' before erroring.
    """
    try:
        # Get libraries, etc from gdal-config (not available on Windows)
        gdal_config = os.environ.get("GDAL_CONFIG", "gdal-config")
        gdal_version_str = read_response([gdal_config, "--version"])

    except Exception as e:
        gdal_version_str = os.environ.get("GDAL_VERSION")

        if not gdal_version_str:
            if platform.system() == "Windows":
                # For Windows: attempt to execute gdalinfo to find GDAL version
                # Note: gdalinfo must be on the PATH
                try:
                    gdalinfo_path = None
                    for path in os.getenv("PATH", "").split(os.pathsep):
                        matches = list(Path(path).glob("**/gdalinfo*"))
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
                        "Could not obtain GDAL version by executing 'gdalinfo --version'"
                    )

        if not gdal_version_str:
            print("GDAL_VERSION must be provided as an environment variable")
            sys.exit(1)

    gdal_version = tuple(int(i) for i in gdal_version_str.strip("dev").split("."))

    if not gdal_version >= MIN_GDAL_VERSION:
        sys.exit(f"GDAL must be >= {'.'.join(map(str, MIN_GDAL_VERSION))}")

    return gdal_version


ext_modules = []
package_data = {}

# setuptools clean does not cleanup Cython artifacts
if "clean" in sys.argv:
    for directory in ["build", "pyogrio/gdal_data", "pyogrio/proj_data"]:
        if os.path.exists(directory):
            shutil.rmtree(directory)

    root = Path(".")
    for ext in ["*.so", "*.pyc", "*.c", "*.cpp"]:
        for entry in root.rglob(ext):
            entry.unlink()

elif "sdist" in sys.argv or "egg_info" in sys.argv:
    # don't cythonize for the sdist
    pass

else:
    if cythonize is None:
        raise ImportError("Cython is required to build from source")

    ext_options = get_gdal_paths()

    compile_time_env = {
        "CTE_GDAL_VERSION": get_gdal_version(),
    }

    ext_modules = cythonize(
        [
            Extension("pyogrio._err", ["pyogrio/_err.pyx"], **ext_options),
            Extension("pyogrio._geometry", ["pyogrio/_geometry.pyx"], **ext_options),
            Extension("pyogrio._io", ["pyogrio/_io.pyx"], **ext_options),
            Extension("pyogrio._ogr", ["pyogrio/_ogr.pyx"], **ext_options),
        ],
        compiler_directives={"language_level": "3"},
        compile_time_env=compile_time_env,
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

    if os.environ.get("PYOGRIO_PACKAGE_DATA"):
        gdal_data = os.environ.get("GDAL_DATA")
        if gdal_data and os.path.exists(gdal_data):
            log.info(f"Copying gdal data from {gdal_data}")
            copy_data_tree(gdal_data, "pyogrio/gdal_data")
        else:
            raise Exception(
                "Could not find GDAL data files for packaging. "
                "Make sure to set the GDAL_DATA environment variable"
            )

        proj_data = os.environ.get("PROJ_LIB")
        if proj_data and os.path.exists(proj_data):
            log.info(f"Copying proj data from {proj_data}")
            copy_data_tree(proj_data, "pyogrio/proj_data")
        else:
            raise Exception(
                "Could not find PROJ data files for packaging. "
                "Make sure to set the PROJ_LIB environment variable"
            )

        package_data = {"pyogrio": ["gdal_data/*", "proj_data/*"]}


version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()
cmdclass["build_ext"] = build_ext

setup(
    name="pyogrio",
    version=version,
    packages=find_packages(),
    url="https://github.com/pyogrio/pyogrio",
    license="MIT",
    author="Brendan C. Ward",
    author_email="bcward@astutespruce.com",
    description="Vectorized spatial vector file format I/O using GDAL/OGR",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    python_requires=">=3.8",
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
    package_data=package_data,
)
