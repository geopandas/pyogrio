API reference
=============

Core
----

.. autofunction:: pyogrio.list_drivers
.. autofunction:: pyogrio.detect_write_driver
.. autofunction:: pyogrio.list_layers
.. autofunction:: pyogrio.read_bounds
.. autofunction:: pyogrio.read_info
.. autofunction:: read_bounds
.. autofunction:: read_info
.. autofunction:: set_gdal_config_options
.. autofunction:: get_gdal_config_option
.. autofunction:: vsi_listtree
.. autofunction:: vsi_rmtree
.. autofunction:: vsi_unlink

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
