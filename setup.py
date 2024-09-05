import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from setuptools import Extension, setup, find_packages
import versioneer

# import Cython if available
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    cythonize = None


logger = logging.getLogger(__name__)


MIN_PYTHON_VERSION = (3, 9, 0)
MIN_GDAL_VERSION = (2, 4, 0)


if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError("Python >= 3.9 is required")


def copy_data_tree(datadir, destdir):
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    shutil.copytree(datadir, destdir)


# Get GDAL config from gdal-config command
def read_response(cmd):
    return subprocess.check_output(cmd).decode("utf").strip()


def get_gdal_config():
    """
    Obtain the paths and version for compiling and linking with the GDAL C-API.

    GDAL_INCLUDE_PATH, GDAL_LIBRARY_PATH, and GDAL_VERSION environment variables
    are used if all are present.

    If those variables are not present, gdal-config is called (it should be
    on the PATH variable). gdal-config provides all the paths and version.

    If no environment variables were specified or gdal-config was not found,
    no additional paths are provided to the extension. It is still possible
    to compile in this case using custom arguments to setup.py.
    """
    include_dir = os.environ.get("GDAL_INCLUDE_PATH")
    library_dir = os.environ.get("GDAL_LIBRARY_PATH")
    gdal_version_str = os.environ.get("GDAL_VERSION")

    if include_dir and library_dir and gdal_version_str:
        gdal_libs = ["gdal"]

        if platform.system() == "Windows":
            # NOTE: if libgdal is built for Windows using CMake, it is now "gdal",
            # but older Windows builds still use "gdal_i"
            if (Path(library_dir) / "gdal_i.lib").exists():
                gdal_libs = ["gdal_i"]

        return {
            "include_dirs": [include_dir],
            "library_dirs": [library_dir],
            "libraries": gdal_libs,
        }, gdal_version_str

    if include_dir or library_dir or gdal_version_str:
        logger.warning(
            "If specifying the GDAL_INCLUDE_PATH, GDAL_LIBRARY_PATH, or GDAL_VERSION "
            "environment variables, you need to specify all of them."
        )

    try:
        # Get libraries, etc from gdal-config (not available on Windows)
        flags = ["cflags", "libs", "version"]
        gdal_config = os.environ.get("GDAL_CONFIG", "gdal-config")
        config = {flag: read_response([gdal_config, f"--{flag}"]) for flag in flags}

        gdal_version_str = config["version"]
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
        }, gdal_version_str

    except Exception as e:
        if platform.system() == "Windows":
            # Get GDAL API version from the command line if specified there.
            if "--gdalversion" in sys.argv:
                index = sys.argv.index("--gdalversion")
                sys.argv.pop(index)
                gdal_version_str = sys.argv.pop(index)
            else:
                print(
                    "GDAL_VERSION must be provided as an environment variable "
                    "or as --gdalversion command line argument"
                )
                sys.exit(1)

            logger.info(
                "Building on Windows requires extra options to setup.py to locate "
                "GDAL files. See the installation documentation."
            )
            return {}, gdal_version_str

        else:
            raise e


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

    ext_options, gdal_version_str = get_gdal_config()

    gdal_version = tuple(int(i) for i in gdal_version_str.strip("dev").split("."))
    if not gdal_version >= MIN_GDAL_VERSION:
        sys.exit(f"GDAL must be >= {'.'.join(map(str, MIN_GDAL_VERSION))}")

    compile_time_env = {
        "CTE_GDAL_VERSION": gdal_version,
    }

    ext_modules = cythonize(
        [
            Extension("pyogrio._err", ["pyogrio/_err.pyx"], **ext_options),
            Extension("pyogrio._geometry", ["pyogrio/_geometry.pyx"], **ext_options),
            Extension("pyogrio._io", ["pyogrio/_io.pyx"], **ext_options),
            Extension("pyogrio._ogr", ["pyogrio/_ogr.pyx"], **ext_options),
            Extension("pyogrio._vsi", ["pyogrio/_vsi.pyx"], **ext_options),
        ],
        compiler_directives={"language_level": "3"},
        compile_time_env=compile_time_env,
    )

    if os.environ.get("PYOGRIO_PACKAGE_DATA"):
        gdal_data = os.environ.get("GDAL_DATA")
        if gdal_data and os.path.exists(gdal_data):
            logger.info(f"Copying gdal data from {gdal_data}")
            copy_data_tree(gdal_data, "pyogrio/gdal_data")
        else:
            raise Exception(
                "Could not find GDAL data files for packaging. "
                "Make sure to set the GDAL_DATA environment variable"
            )

        proj_data = os.environ.get("PROJ_LIB")
        if proj_data and os.path.exists(proj_data):
            logger.info(f"Copying proj data from {proj_data}")
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
    version=version,
    packages=find_packages(),
    include_package_data=True,
    exclude_package_data={'': ['*.h', '_*.pxd', '_*.pyx']},
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    package_data=package_data,
)
