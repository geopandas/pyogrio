API reference
=============

Core
----

.. automodule:: pyogrio
   :members: list_drivers, detect_write_driver, list_layers, read_bounds, read_info, set_gdal_config_options, get_gdal_config_option, __gdal_version__, __gdal_version_string__

GeoPandas integration
---------------------

.. autofunction:: pyogrio.read_dataframe
.. autofunction:: pyogrio.write_dataframe

Reading as Arrow data
---------------------

.. autofunction:: pyogrio.raw.read_arrow
.. autofunction:: pyogrio.raw.open_arrow

Writing from Arrow data
---------------------

.. autofunction:: pyogrio.raw.write_arrow
