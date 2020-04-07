# pyogrio - Vectorized vector I/O using GDAL

Supports Python 3.8 and GDAL 2.4.x

## Goals

### Short term

Stub out enough of a minimal API for reading and writing from GDAL OGR to
establish a rough baseline of how fast vectorized operations like this can be
when built with Cython / Python. This will be used as a comparison of I/O
performance with Fiona to see if the approach can / should be ported there.

### Long term

Provide faster I/O using GDAL OGR to read / write vector geospatial files

Intended for use with libraries that consume WKB for their internal constructs,
such as [pygeos](https://github.com/pygeos/pygeos).

Geometries will be read from sources into ndarray of WKB.
Attributes will be read into ndarrays.

## Installation / development

Clone this repository to a local folder.

Right now, this requires system GDAL 2.4. See `install_extras` in the `setup.py`
for additional dependencies.

Run `python setup.py develop` to build the extensions in Cython.

Test datasets are downloaded and placed into `tests/fixtures/datasets` (each gets its own folder):

[Natural Earth](https://www.naturalearthdata.com/downloads/):

-   [Admin 0 (countries) at 1:110m](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip)
-   [Admin 0 (countries at 1:10m)](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip)
-   [Admin 1 (states / provinces) at 1:110m](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_1_states_provinces.zip)
-   [Admin 1 (states / provinces) at 1:10m](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_1_states_provinces.zip)

Hydrography:

-   [Watershed boundaries](https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GDB/WBD_17_HU2_GDB.zip)
-   [Flowlines, waterbodies, etc](https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_1704_HU4_GDB.zip)

## Examples

### List layers in a vector data source

Layers within the data source are returned as ndarray of strings.

```
>>> pyogrio.list_layers('tests/fixtures/datasets/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
array(['ne_110m_admin_0_countries'], dtype=object)
```

### Read from a vector data layer

By default, the first layer of a given data source is read unless a specific layer is specified by name or index.

3 parts are read from the vector data layer:

-   `meta`: dictionary of high-level information about the data source and layer
-   `geometry`: WKB encoded geometries (if applicable)
-   `field_data`: list of ndarrays of data for each field in the data layer, each field has same shape as all other fields and geometry

#### Meta data structure

```
{
    "crs": <EPSG code or WKT string>,
    "encoding": <"UTF-8" if designated as such by data source, otherwise "ISO-8859-1" if a Shapefile, or determined from the system encoding or passed in by user>,
    "fields": <ndarray of field names>,
    "geometry": <GeoJSON geometry type name, e.g., "Polygon">
}
```

#### Read directly into ndarrays

```
>>> meta, geometry, fields = pyogrio.read('tests/fixtures/datasets/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
>>> meta
{
    'crs': 'EPSG:4326',
    'encoding': 'UTF-8',
    'fields': array([
        'featurecla',
        'scalerank',
        ...<many more>
    ], dtype=object),
    'geometry': 'Polygon'
}
>>> geometry.shape
(177, )
>>> geometry[0]
b'\x01\x06\x00\x00\x00\x03...'
>>> fields[24][:2]
array(['Republic of Fiji', 'United Republic of Tanzania'], dtype=object)
```

#### Read into a pandas DataFrame

Requires `pandas`.

To read into a DataFrame, but keep geometry as WKB:

```
df = pyogrio.read_dataframe('tests/fixtures/datasets/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
```

If you have `pygeos` available, you can convert to `pygeos` geometry objects:

```
df = pyogrio.read_dataframe('tests/fixtures/datasets/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp', as_pygeos=True)
```

### Null values

Some data sources support NULL or otherwise unset field values. These cannot be properly
stored into the ndarray for certain types. If NULL or unset values are encountered,
the following occurs:

-   If the field is a string type, NULL values are represented as None
-   If the field is an integer type (np.int32, np.int64), the field data are re-cast to np.float64 values, and NULL values are represented as np.nan
-   If the field is a date or datetime type, the field is set as np.datetime64('NaT')

### Write to a vector data source

Not implemented yet.

## Performance

Based on initial benchmarks with Natural Earth data and recent versions of `fiona`, `geopandas`, and `pygeos`:

Compared to `fiona`:

-   1.6x faster listing of layers in single-layer data source
-   1.6x - 5x faster reading of small data sources (Natural Earth 10m and 110m Admin 0 and Admin 1 levels)

Compared to `geopandas` in native `shapely` objects, converting data frame here to `pygeos` objects:

-   6.5 - 16.5x faster reading of data into geometry-backed data frames

## Assumptions

Attributes have consistent type across all records.

## WORK IN PROGRESS

This is building toward a proof of concept, it is not meant for production use. Depending on
how this approach works out, it may be refactored into the core of Fiona.

Right now, this borrows heavily from implementations in Fiona.

## Credits

-   Adapted from [fiona](https://github.com/Toblerity/Fiona)
-   Inspired by [fiona PR](https://github.com/Toblerity/Fiona/pull/540/files)
