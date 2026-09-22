# Supported vector formats

All vector formats/drivers available in the local installation of GDAL used
by Pyogrio will be available in Pyogrio as well.

For each format, it depends on the GDAL driver if the format is readonly or
writable. Please see the
[list of GDAL drivers](https://gdal.org/drivers/vector/index.html) for the full
list of possible drivers and more information about them.

You can get a list all drivers available in your local installation with
{func}`pyogrio.list_drivers` or {func}`pyogrio.list_drivers_details`.

## Explicitly supported formats

Most of the functionalities of Pyogrio are tested/verified to work for the
following formats:

-   [ESRI Shapefile](https://gdal.org/drivers/vector/shapefile.html)
-   [GeoPackage](https://gdal.org/drivers/vector/gpkg.html)
-   [GeoJSON](https://gdal.org/drivers/vector/geojson.html) / [GeoJSONSeq](https://gdal.org/drivers/vector/geojsonseq.html)
-   [FlatGeobuf](https://gdal.org/drivers/vector/flatgeobuf.html) (requires GDAL >= 3.1)

You can also use [Virtual File System](https://gdal.org/user/virtual_file_systems.html#virtual-file-systems),
paths to read e.g. remote or zipped data sources and directories.

Most Pyogrio functionalities will most likely work for the other vector formats
available in your local installation of GDAL. Nonetheless, please be aware that
these likely have not been tested for compatibility with Pyogrio and you may
encounter specific issues with these formats and / or their constituent geometry
or field data types.

We are unlikely to support obscure, rarely-used, proprietary vector formats,
especially if they require advanced GDAL installation procedures.
