# Supported vector formats

Support for reading and writing spatial data ultimately depends on what is
available in your particular distribution of GDAL. GDAL supports reading from
a wide number of vector file formats, and writing for a much smaller number.

Please see the [list of drivers](https://gdal.org/drivers/vector/index.html) for
more information.

## Full read and write support

-   [ESRI Shapefile](https://gdal.org/drivers/vector/shapefile.html)
-   [GeoPackage](https://gdal.org/drivers/vector/gpkg.html)
-   [GeoJSON](https://gdal.org/drivers/vector/geojson.html) / [GeoJSONSeq](https://gdal.org/drivers/vector/geojsonseq.html)
-   [FlatGeobuf](https://gdal.org/drivers/vector/flatgeobuf.html) (requires GDAL >= 3.1)

## Read support

-   [ESRI FileGDB (via OpenFileGDB)](https://gdal.org/drivers/vector/openfilegdb.html#vector-openfilegdb)
-   above formats using the [Virtual File System](https://gdal.org/user/virtual_file_systems.html#virtual-file-systems), which supports zipped data sources and directories

## Support for other formats

Other vector formats that are registered within your particular installation of
GDAL may be supported. Please be aware that these likely have not been tested
for compatibility with Pyogrio and you may encounter specific issues with these
formats and / or their constituent geometry or field data types.

We are unlikely to support obscure, rarely-used, proprietary vector formats,
especially if they require advanced GDAL installation procedures.
