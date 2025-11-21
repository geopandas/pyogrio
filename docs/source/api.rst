.. py:currentmodule:: pyogrio

API reference
=============

Core
----

.. automodule:: pyogrio
   :members: list_drivers, detect_write_driver, list_layers, read_bounds, read_info, set_gdal_config_options, get_gdal_config_option, vsi_listtree, vsi_rmtree, vsi_unlink

..
   For the special attributes/dunder attributes, the inline docstrings weren't
   picked up by autodoc, so they are documented explicitly here.

.. py:attribute:: __version__

   The pyogrio version (`str`).

.. py:attribute:: __gdal_version__

   The GDAL version used by pyogrio (`tuple` of `int`).

.. py:attribute:: __gdal_version_string__

   The GDAL version used by pyogrio (`str`).

.. py:attribute:: __gdal_geos_version__

   The version of GEOS used by GDAL (`tuple` of `int`).


GeoPandas integration
---------------------

.. autofunction:: pyogrio.read_dataframe
.. autofunction:: pyogrio.write_dataframe

Arrow integration
-----------------

.. autofunction:: pyogrio.read_arrow
.. autofunction:: pyogrio.open_arrow
.. autofunction:: pyogrio.write_arrow
