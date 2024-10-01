# About

## How it works

Internally, Pyogrio uses a numpy-oriented approach in Cython to read
information about data sources and records from spatial data layers. Geometries
are extracted from the data layer as Well-Known Binary (WKB) objects and fields
(attributes) are read into numpy arrays of the appropriate data type. These are
then converted to GeoPandas `GeoDataFrame`s.

All records are read into memory, which may be problematic for very large data
sources. You can use `skip_features` / `max_features` to read smaller parts of
the file at a time.

The entire `GeoDataFrame` is written at once. Incremental writes or appends to
existing data sources are not supported.

## Comparison to Fiona

[Fiona](https://github.com/Toblerity/Fiona) is a full-featured Python library
for working with OGR vector data sources. It is **awesome**, has highly-dedicated
maintainers and contributors, and exposes more functionality than Pyogrio ever will.
This project would not be possible without Fiona having come first.

Pyogrio uses a bulk-oriented approach for reading and writing
spatial vector file formats, which enables faster I/O operations. It borrows
from the internal mechanics and lessons learned of Fiona. It uses a stateless
approach to reading or writing data; all data are read or written in a single
pass.

`Fiona` is a general-purpose spatial format I/O library that is used within many
projects in the Python ecosystem. In contrast, Pyogrio specifically targets
GeoPandas in order to reduce the number of data transformations currently
required to read and write data between GeoPandas `GeoDataFrame`s and OGR data
sources using Fiona (the current default in GeoPandas).
