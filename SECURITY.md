# Security Policy

Pyogrio wraps [GDAL](https://gdal.org/), which depends on a variety of third-
party dependencies, such as [GEOS](https://libgeos.org/),
[PROJ](https://proj.org/), and [CURL](https://curl.se/). These dependencies
vary based on which features are enabled when GDAL is compiled and vary across
platforms.

Most vulnerabilities are likely to occur in these dependencies.

Please see [GDAL's security policy and published advisories](https://github.com/OSGeo/gdal/security).

If you know that a vulnerability originates from a third-party dependency,
please report the vulnerability directly to the affected dependency.

If the vulnerability requires a modification of Pyogrio's packaging to
specifically avoid linking to a known vulnerable version of a third party
dependency, please report the vulnerability here as well.

## Available packages

Pyogrio is available in 3 basic forms:

### Binary wheels on PyPI

Binary wheels are published to [PyPI](https://pypi.org/project/pyogrio/) and
are created using [VCPKG](https://vcpkg.io/en/) versions of GDAL and associated
dependencies. These wheels include binary libraries of these dependencies.
Because these binary wheels are specifically under our control, these are the
packages where we are most concerned with ensuring that we avoid known
vulnerable versions of dependencies.

### Conda-forge packages

Conda packages are available on
[conda-forge](https://anaconda.org/conda-forge/pyogrio). Pyogrio uses packages
for GDAL and associated dependencies. Please contact the appropriate maintainers
for those packages to report vulnerabilities.

### Self-compiled / local development

When you build Pyogrio locally, you link it to your local version of GDAL and
associated dependencies in a way that depends on how you installed GDAL and its
dependencies (e.g., system package, homebrew package, etc). Please contact the
appropriate maintainers of these packages to report vulnerabilities.

## Supported versions

Pyogrio has not yet reached 1.0. Only the latest available version is being
supported with security updates.

Please see the [releases page](https://github.com/geopandas/pyogrio/releases)
for the latest available release.

## Security advisories

Please see the [security page](https://github.com/geopandas/pyogrio/security)
for published security advisories.

## Reporting a vulnerability

To report a vulnerability in Pyogrio, please use GitHub's "Report a vulnerability"
feature on the [security page](https://github.com/geopandas/pyogrio/security).

### Vulnerabilities in Pyogrio

If the vulnerability is included within Pyogrio source code, please include at
least the following information:

-   location of the vulnerability within the source code (file and expression);
    you can provide a URL to a line or range of lines from within GitHub
-   brief description of the vulnerability sufficient for project maintainers
    to understand the nature of the vulnerability, including conditions that
    will trigger the vulnerable code
-   a small test dataset (if possible) or detailed description of the structure
    and contents of a dataset that will trigger the vulnerability
-   operating system, Python version, and Pyogrio version you were using when
    you detected the vulnerability
-   severity of the dependency: does it pose a critical risk to system or data
    integrity or security, does it pose a high risk for potential loss of data,
    or is it an edge case that poses a minor risk only under specific
    circumstances?

### Vulnerabilities in Pyogrio's dependencies

If the vulnerability is included within Pyogrio's binary wheels from a
third-party dependency or is linked from Pyogrio's conda-forge package, and
would require a specific change in Pyogrio's packaging to avoid linking to a
vulnerable version of the dependency, please include at least the following
information:

-   a link to the published CVE or other description of the vulnerabilty
-   operating system, Python version, and Pyogrio versions that may be impacted
-   if known, the version of the dependency impacted by the vulnerability
-   the package of Pyogrio that is impacted by the vulnerability: binary wheel,
    conda-forge package, etc
-   if known, the range of vulnerable versions of the dependency and the version
    that resolves the vulnerability
