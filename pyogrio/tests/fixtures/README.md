# Test datasets

## Natural Earth lowres

`naturalearth_lowres.shp` was copied from GeoPandas.

## FGDB test dataset

`test_fgdb.gdb.zip`
Downloaded from http://trac.osgeo.org/gdal/raw-attachment/wiki/FileGDB/test_fgdb.gdb.zip

### GPKG test dataset with null values

`test_gpkg_nulls.gpkg` was created using Fiona backend to GeoPandas:

```
from collections import OrderedDict

import fiona
import geopandas as gp
import numpy as np
from pyogrio import write_dataframe

filename = "test_gpkg_nulls.gpkg"

df = gp.GeoDataFrame(
    {
        "col_bool": np.array([True, False, True], dtype="bool"),
        "col_int8": np.array([1, 2, 3], dtype="int8"),
        "col_int16": np.array([1, 2, 3], dtype="int16"),
        "col_int32": np.array([1, 2, 3], dtype="int32"),
        "col_int64": np.array([1, 2, 3], dtype="int64"),
        "col_uint8": np.array([1, 2, 3], dtype="uint8"),
        "col_uint16": np.array([1, 2, 3], dtype="uint16"),
        "col_uint32": np.array([1, 2, 3], dtype="uint32"),
        "col_uint64": np.array([1, 2, 3], dtype="uint64"),
        "col_float32": np.array([1.5, 2.5, 3.5], dtype="float32"),
        "col_float64": np.array([1.5, 2.5, 3.5], dtype="float64"),
    },
    geometry=gp.points_from_xy([0, 1, 2], [0, 1, 2]),
    crs="EPSG:4326",
)

write_dataframe(df, filename)

# construct row with null values
# Note: np.nan can only be used for float values
null_row = {
    "type": "Fetaure",
    "id": 4,
    "properties": OrderedDict(
        [
            ("col_bool", None),
            ("col_int8", None),
            ("col_int16", None),
            ("col_int32", None),
            ("col_int64", None),
            ("col_uint8", None),
            ("col_uint16", None),
            ("col_uint32", None),
            ("col_uint64", None),
            ("col_float32", np.nan),
            ("col_float64", np.nan),
        ]
    ),
    "geometry": {"type": "Point", "coordinates": (4.0, 4.0)},
}

# append row with nulls to GPKG
with fiona.open(filename, "a") as c:
    c.write(null_row)
```

NOTE: Reading boolean values into GeoPandas using Fiona backend treats those
values as `None` and column dtype as `object`; Pyogrio treats those values as
`np.nan` and column dtype as `float64`.

### GPKG test with MultiSurface

This was extracted from https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_0308_HU4_GDB.zip
`NHDWaterbody` layer using ogr2ogr:

```bash
ogr2ogr test_mixed_surface.gpkg NHDPLUS_H_0308_HU4_GDB.gdb NHDWaterbody -where '"NHDPlusID" = 15000300070477' -select "NHDPlusID"
```

### OSM PBF test

This was downloaded from https://github.com/openstreetmap/OSM-binary/blob/master/resources/sample.pbf
