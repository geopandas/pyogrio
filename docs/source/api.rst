API reference
=============

Core
----

.. automodule:: pyogrio
   :members: list_drivers, detect_write_driver, list_layers, read_bounds, read_info, set_gdal_config_options, get_gdal_config_option, vsi_listtree, vsi_rmtree, vsi_unlink

.. autoproperty:: pyogrio.__version__
.. autoproperty:: pyogrio.__gdal_version__
.. autoproperty:: pyogrio.__gdal_version_string__

GeoPandas integration
---------------------

.. autofunction:: pyogrio.read_dataframe
.. autofunction:: pyogrio.write_dataframe

Arrow integration
-----------------

.. autofunction:: pyogrio.read_arrow
.. autofunction:: pyogrio.open_arrow
.. autofunction:: pyogrio.write_arrow
