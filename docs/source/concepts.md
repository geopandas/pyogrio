# Concepts and Terminology

## GDAL / OGR

Pyogrio is powered by [GDAL/OGR](https://gdal.org/index.html). OGR is the part
of the GDAL library that specifically provides interoperability with vector data
sources. Vector data sources are those that contain geometries (points, lines,
or polygons) and associated columns of data.

We refer to GDAL / OGR interchangeably throughout the documentation.

## OGR vector data source

An OGR vector data source is a container file format, it may contain one or
several spatial and / or nonspatial data layers or tables depending on its type.

For example, a GeoPackage may contain several spatial data layers. In contrast,
an ESRI Shapefile always consists of a single data layer.

## OGR vector data layer

An OGR vector data layer is a single entity within a vector data source, and may
have 0 or more records and may or may not include a geometry column, depending
on the data layer type.

## OGR vector driver

An OGR vector driver is implemented directly within the GDAl / OGR library, and
is what ultimately provides the ability to read or write a specific vector data
source format. GDAL is typically distributed with drivers enabled for some of
the most common vector formats, whereas others are opt-in and included only
within specific distributions of GDAL or if you compile it yourself.

See the [list of drivers](https://gdal.org/drivers/vector/index.html) for
more information.
