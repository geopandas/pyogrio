# Updating wheel builds to a new GDAL release

This is a short runbook for bumping the GDAL version bundled in pyogrio wheels.

It is based on the changes made in:
- https://github.com/geopandas/pyogrio/pull/578
- https://github.com/geopandas/pyogrio/pull/591

## Wheel build setup

Wheel building is orchestrated by `.github/workflows/release.yml` using
`cibuildwheel`.

The hard part, building GDAL to be vendored in the wheel for all platforms, is
done using [vcpkg](https://vcpkg.io/en/).

Then specifically:

- Linux wheels are built inside custom manylinux Docker images in `ci/`.
  Those use the upstream PyPA manylinux images as a base with additional layers
  to have GDAL installed (those images get cached to avoid rebuilding each time).
- macOS and Windows wheels install GDAL directly with vcpkg in the GitHub
  Actions runner image.

For building a specific GDAL version reproducibly (with same set of dependencies,
until the GDAL version is bumped), we pin the vcpgk baseline to a specific commit
(specified in `ci/vcpkg.json` and as the `VCPKG_GDAL_COMMIT` env variable which
is used to checkout that exact baseline of vcpkg).

The exact GDAL features that are enabled are defined in `ci/vcpkg.json.

Some vcpkg build customizations:

- Custom overlay triplets (defined in `ci/custom-triplets/`): defined for each
  platform to consistently use a Release build and a GDAL dynamic shared library
  (with static linking for all transitive dependencies)
- Custom overlay ports: for zlib, as we needed an older version than what is
  used in the baseline (manylinux considers this as a system dependency, and if
  built with a newer version, auditwheel complains) -> only needed for
  manylinux2014.

## Update procedure

### 1) Pick the target versions

You need two values:

- `GDAL_VERSION`: the target GDAL release (for example `3.11.5`)
- `VCPKG_GDAL_COMMIT`: a vcpkg commit that provides that GDAL version

Notes:
- Use a vcpkg commit known to include the target GDAL port update. Typically,
  search for the vcpkg commit that last updated the GDAL port (occasionally
  we need a more recent commit if a build fix for one of the dependencies is
  needed).
- In practice, it is the vcpkg commit that will determine the exact GDAL version
  being built (i.e. the version of the GDAL port at that commit), the
  `GDAL_VERSION` env variable is purely informative (and to flush the cache).
  But ensure to keep `GDAL_VERSION` and `VCPKG_GDAL_COMMIT` in sync.

### 2) Update the release workflow knobs

Edit `.github/workflows/release.yml`:

- Update top-level `env` values:
  - `GDAL_VERSION`
  - `VCPKG_GDAL_COMMIT`
- Update the `test-sdist.container.image` to use the same GDAL version

This workflow is already parameterized to use `VCPKG_GDAL_COMMIT` for ensuring
the same vcpkg baseline in the Linux Docker image build args and macOS/Windows
vcpkg checkout.

### 3) Update vcpkg.json baselines

Edit:
- `ci/vcpkg.json`
- `ci/vcpkg-manylinux2014.json`

In both files, set:
- `builtin-baseline` to the same as `VCPKG_GDAL_COMMIT`

### 4) Update the manylinux base image (if needed)

Occasionally, the base for the Linux docker images needs to be updated:

```
FROM quay.io/pypa/manylinux_2_28_x86_64:2025.09.19-1
```

Update to the latest named (date-based) tag available (see e.g. https://quay.io/repository/pypa/manylinux_2_28_x86_64?tab=tags).

This is needed, for example, when a new Python version needs to be supported
(in that case, also add this Python version to the `test-wheels` job in
`release.yml`).

When there is a vcpkg build failure for GDAL or one of its dependencies, updating
the base image is also a first thing to try.

### 5) Changelog note

Add or update the packaging note in `CHANGES.md` to record the GDAL bump.

> The GDAL library included in the wheels is upgraded from X to Y (#xxx).

### 6) Open PR and validate CI

Open a PR:
- Suggested title: `BLD/RLS: update wheels to include GDAL X.Y.Z`
- PR body mentioning the updated vcpkg commit: `Updating to GDAL X.Y.Z (https://github.com/microsoft/vcpkg/commit/<VCPKG_GDAL_COMMIT>).`

For debugging vcpkg build failures:
- For macOS and Windows, the vcpkg build logs are uploaded as job artifact in
  case of a failure
- For linux, you can build the docker container locally up to the vcpkg install step, and then run the container and manually run the vcpkg install step, and inspect the logs
