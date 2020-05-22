# things queued for porting from fiona


cdef class Session:

    cdef void *cogr_ds
    cdef void *cogr_layer
    cdef object _fileencoding
    cdef object _encoding
    cdef object collection

    def __init__(self):
        self.cogr_ds = NULL
        self.cogr_layer = NULL
        self._fileencoding = None
        self._encoding = None

    def __dealloc__(self):
        self.stop()

    def start(self, collection, **kwargs):
        cdef const char *path_c = NULL
        cdef const char *name_c = NULL
        cdef void *drv = NULL
        cdef void *ds = NULL
        cdef char **ignore_fields = NULL

        path_b = collection.path.encode('utf-8')
        path_c = path_b

        self._fileencoding = kwargs.get('encoding') or collection.encoding

        # We have two ways of specifying drivers to try. Resolve the
        # values into a single set of driver short names.
        if collection._driver:
            drivers = set([collection._driver])
        elif collection.enabled_drivers:
            drivers = set(collection.enabled_drivers)
        else:
            drivers = None

        encoding = kwargs.pop('encoding', None)
        if encoding:
            kwargs['encoding'] = encoding.upper()

        self.cogr_ds = gdal_open_vector(path_c, 0, drivers, kwargs)

        if isinstance(collection.name, string_types):
            name_b = collection.name.encode('utf-8')
            name_c = name_b
            self.cogr_layer = GDALDatasetGetLayerByName(self.cogr_ds, name_c)
        elif isinstance(collection.name, int):
            self.cogr_layer = GDALDatasetGetLayer(self.cogr_ds, collection.name)
            name_c = OGR_L_GetName(self.cogr_layer)
            name_b = name_c
            collection.name = name_b.decode('utf-8')

        if self.cogr_layer == NULL:
            raise ValueError("Null layer: " + repr(collection.name))

        encoding = self._get_internal_encoding()

        if collection.ignore_fields:
            try:
                for name in collection.ignore_fields:
                    try:
                        name_b = name.encode(encoding)
                    except AttributeError:
                        raise TypeError("Ignored field \"{}\" has type \"{}\", expected string".format(name, name.__class__.__name__))
                    ignore_fields = CSLAddString(ignore_fields, <const char *>name_b)
                OGR_L_SetIgnoredFields(self.cogr_layer, <const char**>ignore_fields)
            finally:
                CSLDestroy(ignore_fields)

        self.collection = collection

    cpdef stop(self):
        self.cogr_layer = NULL
        if self.cogr_ds != NULL:
            GDALClose(self.cogr_ds)
        self.cogr_ds = NULL

    def get_fileencoding(self):
        """DEPRECATED"""
        warnings.warn("get_fileencoding is deprecated and will be removed in a future version.", FionaDeprecationWarning)
        return self._fileencoding

    def _get_fallback_encoding(self):
        """Determine a format-specific fallback encoding to use when using OGR_F functions
        Parameters
        ----------
        None
        Returns
        -------
        str
        """
        if "Shapefile" in self.get_driver():
            return 'iso-8859-1'
        else:
            return locale.getpreferredencoding()


    def _get_internal_encoding(self):
        """Determine the encoding to use when use OGR_F functions
        Parameters
        ----------
        None
        Returns
        -------
        str
        Notes
        -----
        If the layer implements RFC 23 support for UTF-8, the return
        value will be 'utf-8' and callers can be certain that this is
        correct.  If the layer does not have the OLC_STRINGSASUTF8
        capability marker, it is not possible to know exactly what the
        internal encoding is and this method returns best guesses. That
        means ISO-8859-1 for shapefiles and the locale's preferred
        encoding for other formats such as CSV files.
        """
        if OGR_L_TestCapability(self.cogr_layer, OLC_STRINGSASUTF8):
            return 'utf-8'
        else:
            return self._fileencoding or self._get_fallback_encoding()

    def get_length(self):
        if self.cogr_layer == NULL:
            raise ValueError("Null layer")
        return OGR_L_GetFeatureCount(self.cogr_layer, 0)

    def get_driver(self):
        cdef void *cogr_driver = GDALGetDatasetDriver(self.cogr_ds)
        if cogr_driver == NULL:
            raise ValueError("Null driver")
        cdef const char *name = OGR_Dr_GetName(cogr_driver)
        driver_name = name
        return driver_name.decode()

    def get_schema(self):
        cdef int i
        cdef int n
        cdef void *cogr_featuredefn = NULL
        cdef void *cogr_fielddefn = NULL
        cdef const char *key_c
        props = []

        if self.cogr_layer == NULL:
            raise ValueError("Null layer")

        if self.collection.ignore_fields:
            ignore_fields = self.collection.ignore_fields
        else:
            ignore_fields = set()

        cogr_featuredefn = OGR_L_GetLayerDefn(self.cogr_layer)
        if cogr_featuredefn == NULL:
            raise ValueError("Null feature definition")

        encoding = self._get_internal_encoding()

        n = OGR_FD_GetFieldCount(cogr_featuredefn)

        for i from 0 <= i < n:
            cogr_fielddefn = OGR_FD_GetFieldDefn(cogr_featuredefn, i)
            if cogr_fielddefn == NULL:
                raise ValueError("Null field definition")

            key_c = OGR_Fld_GetNameRef(cogr_fielddefn)
            key_b = key_c

            if not bool(key_b):
                raise ValueError("Invalid field name ref: %s" % key)

            key = key_b.decode(encoding)

            if key in ignore_fields:
                continue

            fieldtypename = FIELD_TYPES[OGR_Fld_GetType(cogr_fielddefn)]
            if not fieldtypename:
                log.warning(
                    "Skipping field %s: invalid type %s",
                    key,
                    OGR_Fld_GetType(cogr_fielddefn))
                continue

            val = fieldtypename
            if fieldtypename == 'float':
                fmt = ""
                width = OGR_Fld_GetWidth(cogr_fielddefn)
                if width: # and width != 24:
                    fmt = ":%d" % width
                precision = OGR_Fld_GetPrecision(cogr_fielddefn)
                if precision: # and precision != 15:
                    fmt += ".%d" % precision
                val = "float" + fmt
            elif fieldtypename in ('int32', 'int64'):
                fmt = ""
                width = OGR_Fld_GetWidth(cogr_fielddefn)
                if width:
                    fmt = ":%d" % width
                val = 'int' + fmt
            elif fieldtypename == 'str':
                fmt = ""
                width = OGR_Fld_GetWidth(cogr_fielddefn)
                if width:
                    fmt = ":%d" % width
                val = fieldtypename + fmt

            props.append((key, val))

        ret = {"properties": OrderedDict(props)}

        if not self.collection.ignore_geometry:
            code = normalize_geometry_type_code(
                OGR_FD_GetGeomType(cogr_featuredefn))
            ret["geometry"] = GEOMETRY_TYPES[code]

        return ret

    def get_crs(self):
        """Get the layer's CRS
        Returns
        -------
        CRS
        """
        cdef char *proj_c = NULL
        cdef const char *auth_key = NULL
        cdef const char *auth_val = NULL
        cdef void *cogr_crs = NULL

        if self.cogr_layer == NULL:
            raise ValueError("Null layer")

        try:
            cogr_crs = exc_wrap_pointer(OGR_L_GetSpatialRef(self.cogr_layer))
        # TODO: we don't intend to use try/except for flow control
        # this is a work around for a GDAL issue.
        except FionaNullPointerError:
            log.debug("Layer has no coordinate system")

        if cogr_crs is not NULL:

            log.debug("Got coordinate system")
            crs = {}

            try:

                retval = OSRAutoIdentifyEPSG(cogr_crs)
                if retval > 0:
                    log.info("Failed to auto identify EPSG: %d", retval)

                try:
                    auth_key = <const char *>exc_wrap_pointer(<void *>OSRGetAuthorityName(cogr_crs, NULL))
                    auth_val = <const char *>exc_wrap_pointer(<void *>OSRGetAuthorityCode(cogr_crs, NULL))

                except CPLE_BaseError as exc:
                    log.debug("{}".format(exc))

                if auth_key != NULL and auth_val != NULL:
                    key_b = auth_key
                    key = key_b.decode('utf-8')
                    if key == 'EPSG':
                        val_b = auth_val
                        val = val_b.decode('utf-8')
                        crs['init'] = "epsg:" + val

                else:
                    OSRExportToProj4(cogr_crs, &proj_c)
                    if proj_c == NULL:
                        raise ValueError("Null projection")
                    proj_b = proj_c
                    log.debug("Params: %s", proj_b)
                    value = proj_b.decode()
                    value = value.strip()
                    for param in value.split():
                        kv = param.split("=")
                        if len(kv) == 2:
                            k, v = kv
                            try:
                                v = float(v)
                                if v % 1 == 0:
                                    v = int(v)
                            except ValueError:
                                # Leave v as a string
                                pass
                        elif len(kv) == 1:
                            k, v = kv[0], True
                        else:
                            raise ValueError("Unexpected proj parameter %s" % param)
                        k = k.lstrip("+")
                        crs[k] = v

            finally:
                CPLFree(proj_c)
                return crs

        else:
            log.debug("Projection not found (cogr_crs was NULL)")

        return {}

    def get_crs_wkt(self):
        cdef char *proj_c = NULL
        cdef void *cogr_crs = NULL

        if self.cogr_layer == NULL:
            raise ValueError("Null layer")

        try:
            cogr_crs = exc_wrap_pointer(OGR_L_GetSpatialRef(self.cogr_layer))

        # TODO: we don't intend to use try/except for flow control
        # this is a work around for a GDAL issue.
        except FionaNullPointerError:
            log.debug("Layer has no coordinate system")
        except fiona._err.CPLE_OpenFailedError as exc:
            log.debug("A support file wasn't opened. See the preceding ERROR level message.")
            cogr_crs = OGR_L_GetSpatialRef(self.cogr_layer)
            log.debug("Called OGR_L_GetSpatialRef() again without error checking.")
            if cogr_crs == NULL:
                raise exc

        if cogr_crs is not NULL:
            log.debug("Got coordinate system")

            try:
                OSRExportToWkt(cogr_crs, &proj_c)
                if proj_c == NULL:
                    raise ValueError("Null projection")
                proj_b = proj_c
                crs_wkt = proj_b.decode('utf-8')

            finally:
                CPLFree(proj_c)
                return crs_wkt

        else:
            log.debug("Projection not found (cogr_crs was NULL)")
            return ""

    def get_extent(self):
        cdef OGREnvelope extent

        if self.cogr_layer == NULL:
            raise ValueError("Null layer")

        result = OGR_L_GetExtent(self.cogr_layer, &extent, 1)
        return (extent.MinX, extent.MinY, extent.MaxX, extent.MaxY)

    def has_feature(self, fid):
        """Provides access to feature data by FID.
        Supports Collection.__contains__().
        """
        cdef void * cogr_feature
        fid = int(fid)
        cogr_feature = OGR_L_GetFeature(self.cogr_layer, fid)
        if cogr_feature != NULL:
            _deleteOgrFeature(cogr_feature)
            return True
        else:
            return False

    def get_feature(self, fid):
        """Provides access to feature data by FID.
        Supports Collection.__contains__().
        """
        cdef void * cogr_feature
        fid = int(fid)
        cogr_feature = OGR_L_GetFeature(self.cogr_layer, fid)
        if cogr_feature != NULL:
            feature = FeatureBuilder().build(
                cogr_feature,
                encoding=self._get_internal_encoding(),
                bbox=False,
                driver=self.collection.driver,
                ignore_fields=self.collection.ignore_fields,
                ignore_geometry=self.collection.ignore_geometry,
            )
            _deleteOgrFeature(cogr_feature)
            return feature
        else:
            raise KeyError("There is no feature with fid {!r}".format(fid))

    get = get_feature

    # TODO: Make this an alias for get_feature in a future version.
    def __getitem__(self, item):
        cdef void * cogr_feature
        if isinstance(item, slice):
            warnings.warn("Collection slicing is deprecated and will be disabled in a future version.", FionaDeprecationWarning)
            itr = Iterator(self.collection, item.start, item.stop, item.step)
            return list(itr)
        elif isinstance(item, int):
            index = item
            # from the back
            if index < 0:
                ftcount = OGR_L_GetFeatureCount(self.cogr_layer, 0)
                if ftcount == -1:
                    raise IndexError(
                        "collection's dataset does not support negative indexes")
                index += ftcount
            cogr_feature = OGR_L_GetFeature(self.cogr_layer, index)
            if cogr_feature == NULL:
                return None
            feature = FeatureBuilder().build(
                cogr_feature,
                encoding=self._get_internal_encoding(),
                bbox=False,
                driver=self.collection.driver,
                ignore_fields=self.collection.ignore_fields,
                ignore_geometry=self.collection.ignore_geometry,
            )
            _deleteOgrFeature(cogr_feature)
            return feature

    def isactive(self):
        if self.cogr_layer != NULL and self.cogr_ds != NULL:
            return 1
        else:
            return 0


cdef class WritingSession(Session):

    cdef object _schema_mapping

    def start(self, collection, **kwargs):
        cdef OGRSpatialReferenceH cogr_srs = NULL
        cdef char * const *options = NULL
        cdef const char *path_c = NULL
        cdef const char *driver_c = NULL
        cdef const char *name_c = NULL
        cdef const char *proj_c = NULL
        cdef const char *fileencoding_c = NULL
        cdef OGRFieldSubType field_subtype
        cdef int ret
        path = collection.path
        self.collection = collection

        userencoding = kwargs.get('encoding')

        if collection.mode == 'a':

            if not os.path.exists(path):
                raise OSError("No such file or directory %s" % path)
            path_b = strencode(path)
            path_c = path_b

            try:
                self.cogr_ds = gdal_open_vector(path_c, 1, None, kwargs)

                if isinstance(collection.name, string_types):
                    name_b = collection.name.encode('utf-8')
                    name_c = name_b
                    self.cogr_layer = exc_wrap_pointer(GDALDatasetGetLayerByName(self.cogr_ds, name_c))

                elif isinstance(collection.name, int):
                    self.cogr_layer = exc_wrap_pointer(GDALDatasetGetLayer(self.cogr_ds, collection.name))

            except CPLE_BaseError as exc:
                OGRReleaseDataSource(self.cogr_ds)
                self.cogr_ds = NULL
                self.cogr_layer = NULL
                raise DriverError(u"{}".format(exc))

            else:
                self._fileencoding = userencoding or self._get_fallback_encoding()

        elif collection.mode == 'w':
            path_b = strencode(path)
            path_c = path_b

            driver_b = collection.driver.encode()
            driver_c = driver_b
            cogr_driver = exc_wrap_pointer(GDALGetDriverByName(driver_c))

            # Our most common use case is the creation of a new data
            # file and historically we've assumed that it's a file on
            # the local filesystem and queryable via os.path.
            #
            # TODO: remove the assumption.
            if not os.path.exists(path):
                log.debug("File doesn't exist. Creating a new one...")
                cogr_ds = gdal_create(cogr_driver, path_c, {})

            # TODO: revisit the logic in the following blocks when we
            # change the assumption above.
            else:
                if collection.driver == "GeoJSON" and os.path.exists(path):
                    # manually remove geojson file as GDAL doesn't do this for us
                    os.unlink(path)
                try:
                    # attempt to open existing dataset in write mode
                    cogr_ds = gdal_open_vector(path_c, 1, None, kwargs)
                except DriverError:
                    # failed, attempt to create it
                    cogr_ds = gdal_create(cogr_driver, path_c, kwargs)
                else:
                    # check capability of creating a new layer in the existing dataset
                    capability = check_capability_create_layer(cogr_ds)
                    if GDAL_VERSION_NUM < 2000000 and collection.driver == "GeoJSON":
                        # GeoJSON driver tells lies about it's capability
                        capability = False
                    if not capability or collection.name is None:
                        # unable to use existing dataset, recreate it
                        GDALClose(cogr_ds)
                        cogr_ds = NULL
                        cogr_ds = gdal_create(cogr_driver, path_c, kwargs)

            self.cogr_ds = cogr_ds

            # Set the spatial reference system from the crs given to the
            # collection constructor. We by-pass the crs_wkt
            # properties because they aren't accessible until the layer
            # is constructed (later).
            try:
                col_crs = collection._crs_wkt
                if col_crs:
                    cogr_srs = exc_wrap_pointer(OSRNewSpatialReference(NULL))
                    proj_b = col_crs.encode('utf-8')
                    proj_c = proj_b
                    OSRSetFromUserInput(cogr_srs, proj_c)
                    osr_set_traditional_axis_mapping_strategy(cogr_srs)
            except CPLE_BaseError as exc:
                OGRReleaseDataSource(self.cogr_ds)
                self.cogr_ds = NULL
                self.cogr_layer = NULL
                raise CRSError(u"{}".format(exc))

            # Determine which encoding to use. The encoding parameter given to
            # the collection constructor takes highest precedence, then
            # 'iso-8859-1' (for shapefiles), then the system's default encoding
            # as last resort.
            sysencoding = locale.getpreferredencoding()
            self._fileencoding = userencoding or ("Shapefile" in collection.driver and 'iso-8859-1') or sysencoding

            if "Shapefile" in collection.driver:
                if self._fileencoding:
                    fileencoding_b = self._fileencoding.upper().encode('utf-8')
                    fileencoding_c = fileencoding_b
                    options = CSLSetNameValue(options, "ENCODING", fileencoding_c)

            # Does the layer exist already? If so, we delete it.
            layer_count = GDALDatasetGetLayerCount(self.cogr_ds)
            layer_names = []
            for i in range(layer_count):
                cogr_layer = GDALDatasetGetLayer(cogr_ds, i)
                name_c = OGR_L_GetName(cogr_layer)
                name_b = name_c
                layer_names.append(name_b.decode('utf-8'))

            idx = -1
            if isinstance(collection.name, string_types):
                if collection.name in layer_names:
                    idx = layer_names.index(collection.name)
            elif isinstance(collection.name, int):
                if collection.name >= 0 and collection.name < layer_count:
                    idx = collection.name
            if idx >= 0:
                log.debug("Deleted pre-existing layer at %s", collection.name)
                GDALDatasetDeleteLayer(self.cogr_ds, idx)

            # Create the named layer in the datasource.
            name_b = collection.name.encode('utf-8')
            name_c = name_b

            for k, v in kwargs.items():

                if v is None:
                    continue

                # We need to remove encoding from the layer creation
                # options if we're not creating a shapefile.
                if k == 'encoding' and "Shapefile" not in collection.driver:
                    continue

                k = k.upper().encode('utf-8')

                if isinstance(v, bool):
                    v = ('ON' if v else 'OFF').encode('utf-8')
                else:
                    v = str(v).encode('utf-8')
                log.debug("Set option %r: %r", k, v)
                options = CSLAddNameValue(options, <const char *>k, <const char *>v)

            geometry_type = collection.schema.get("geometry", "Unknown")
            if not isinstance(geometry_type, string_types) and geometry_type is not None:
                geometry_types = set(geometry_type)
                if len(geometry_types) > 1:
                    geometry_type = "Unknown"
                else:
                    geometry_type = geometry_types.pop()
            if geometry_type == "Any" or geometry_type is None:
                geometry_type = "Unknown"
            geometry_code = geometry_type_code(geometry_type)

            try:
                self.cogr_layer = exc_wrap_pointer(
                    GDALDatasetCreateLayer(
                        self.cogr_ds, name_c, cogr_srs,
                        <OGRwkbGeometryType>geometry_code, options))

            except Exception as exc:
                OGRReleaseDataSource(self.cogr_ds)
                self.cogr_ds = NULL
                raise DriverIOError(u"{}".format(exc))

            finally:
                if options != NULL:
                    CSLDestroy(options)

                # Shapefile layers make a copy of the passed srs. GPKG
                # layers, on the other hand, increment its reference
                # count. OSRRelease() is the safe way to release
                # OGRSpatialReferenceH.
                if cogr_srs != NULL:
                    OSRRelease(cogr_srs)

            log.debug("Created layer %s", collection.name)

            # Next, make a layer definition from the given schema properties,
            # which are an ordered dict since Fiona 1.0.1.

            encoding = self._get_internal_encoding()

            for key, value in collection.schema['properties'].items():

                log.debug("Begin creating field: %r value: %r", key, value)

                field_subtype = OFSTNone

                # Convert 'long' to 'int'. See
                # https://github.com/Toblerity/Fiona/issues/101.
                if fiona.gdal_version.major >= 2 and value in ('int', 'long'):
                    value = 'int64'
                elif value == 'int':
                    value = 'int32'

                if value == 'bool':
                    value = 'int32'
                    field_subtype = OFSTBoolean

                # Is there a field width/precision?
                width = precision = None
                if ':' in value:
                    value, fmt = value.split(':')

                    log.debug("Field format parsing, value: %r, fmt: %r", value, fmt)

                    if '.' in fmt:
                        width, precision = map(int, fmt.split('.'))
                    else:
                        width = int(fmt)

                    if value == 'int':
                        if GDAL_VERSION_NUM >= 2000000 and (width == 0 or width >= 10):
                            value = 'int64'
                        else:
                            value = 'int32'

                field_type = FIELD_TYPES.index(value)

                try:
                    key_bytes = key.encode(encoding)
                    cogr_fielddefn = exc_wrap_pointer(OGR_Fld_Create(key_bytes, <OGRFieldType>field_type))
                    if width:
                        OGR_Fld_SetWidth(cogr_fielddefn, width)
                    if precision:
                        OGR_Fld_SetPrecision(cogr_fielddefn, precision)
                    if field_subtype != OFSTNone:
                        # subtypes are new in GDAL 2.x, ignored in 1.x
                        set_field_subtype(cogr_fielddefn, field_subtype)
                    exc_wrap_int(OGR_L_CreateField(self.cogr_layer, cogr_fielddefn, 1))

                except (UnicodeEncodeError, CPLE_BaseError) as exc:
                    OGRReleaseDataSource(self.cogr_ds)
                    self.cogr_ds = NULL
                    self.cogr_layer = NULL
                    raise SchemaError(u"{}".format(exc))

                else:
                    OGR_Fld_Destroy(cogr_fielddefn)
                    log.debug("End creating field %r", key)

        # Mapping of the Python collection schema to the munged
        # OGR schema.
        ogr_schema = self.get_schema()
        self._schema_mapping = dict(zip(
            collection.schema['properties'].keys(),
            ogr_schema['properties'].keys() ))

        log.debug("Writing started")

    def writerecs(self, records, collection):
        """Writes buffered records to OGR."""
        cdef void *cogr_driver
        cdef void *cogr_feature
        cdef int features_in_transaction = 0

        cdef void *cogr_layer = self.cogr_layer
        if cogr_layer == NULL:
            raise ValueError("Null layer")

        schema_geom_type = collection.schema['geometry']
        cogr_driver = GDALGetDatasetDriver(self.cogr_ds)
        driver_name = OGR_Dr_GetName(cogr_driver).decode("utf-8")

        valid_geom_types = collection._valid_geom_types
        def validate_geometry_type(record):
            if record["geometry"] is None:
                return True
            return record["geometry"]["type"].lstrip("3D ") in valid_geom_types

        log.debug("Starting transaction (initial)")
        result = gdal_start_transaction(self.cogr_ds, 0)
        if result == OGRERR_FAILURE:
            raise TransactionError("Failed to start transaction")

        schema_props_keys = set(collection.schema['properties'].keys())
        for record in records:
            if not validate_geometry_type(record):
                raise GeometryTypeValidationError(
                    "Record's geometry type does not match "
                    "collection schema's geometry type: %r != %r" % (
                        record['geometry']['type'],
                        collection.schema['geometry'] ))
            # Validate against collection's schema to give useful message
            if set(record['properties'].keys()) != schema_props_keys:
                raise SchemaError(
                    "Record does not match collection schema: %r != %r" % (
                        record['properties'].keys(),
                        list(schema_props_keys) ))
            cogr_feature = OGRFeatureBuilder().build(record, collection)
            result = OGR_L_CreateFeature(cogr_layer, cogr_feature)
            if result != OGRERR_NONE:
                raise RuntimeError("Failed to write record: %s" % record)
            _deleteOgrFeature(cogr_feature)

            features_in_transaction += 1
            if features_in_transaction == DEFAULT_TRANSACTION_SIZE:
                log.debug("Committing transaction (intermediate)")
                result = gdal_commit_transaction(self.cogr_ds)
                if result == OGRERR_FAILURE:
                    raise TransactionError("Failed to commit transaction")
                log.debug("Starting transaction (intermediate)")
                result = gdal_start_transaction(self.cogr_ds, 0)
                if result == OGRERR_FAILURE:
                    raise TransactionError("Failed to start transaction")
                features_in_transaction = 0

        log.debug("Committing transaction (final)")
        result = gdal_commit_transaction(self.cogr_ds)
        if result == OGRERR_FAILURE:
            raise TransactionError("Failed to commit transaction")

    def sync(self, collection):
        """Syncs OGR to disk."""
        cdef void *cogr_ds = self.cogr_ds
        cdef void *cogr_layer = self.cogr_layer
        if cogr_ds == NULL:
            raise ValueError("Null data source")


        gdal_flush_cache(cogr_ds)
        log.debug("Flushed data source cache")

cdef class OGRFeatureBuilder:

    """Builds an OGR Feature from a Fiona feature mapping.
    Allocates one OGR Feature which should be destroyed by the caller.
    Borrows a layer definition from the collection.
    """

    cdef void * build(self, feature, collection) except NULL:
        cdef void *cogr_geometry = NULL
        cdef const char *string_c = NULL
        cdef WritingSession session
        session = collection.session
        cdef void *cogr_layer = session.cogr_layer
        if cogr_layer == NULL:
            raise ValueError("Null layer")
        cdef void *cogr_featuredefn = OGR_L_GetLayerDefn(cogr_layer)
        if cogr_featuredefn == NULL:
            raise ValueError("Null feature definition")
        cdef void *cogr_feature = OGR_F_Create(cogr_featuredefn)
        if cogr_feature == NULL:
            raise ValueError("Null feature")

        if feature['geometry'] is not None:
            cogr_geometry = OGRGeomBuilder().build(
                                feature['geometry'])
        OGR_F_SetGeometryDirectly(cogr_feature, cogr_geometry)

        # OGR_F_SetFieldString takes encoded strings ('bytes' in Python 3).
        encoding = session._get_internal_encoding()

        for key, value in feature['properties'].items():
            ogr_key = session._schema_mapping[key]

            schema_type = normalize_field_type(collection.schema['properties'][key])

            key_bytes = strencode(ogr_key, encoding)
            key_c = key_bytes
            i = OGR_F_GetFieldIndex(cogr_feature, key_c)
            if i < 0:
                continue

            # Special case: serialize dicts to assist OGR.
            if isinstance(value, dict):
                value = json.dumps(value)

            # Continue over the standard OGR types.
            if isinstance(value, integer_types):
                if schema_type == 'int32':
                    OGR_F_SetFieldInteger(cogr_feature, i, value)
                else:
                    OGR_F_SetFieldInteger64(cogr_feature, i, value)

            elif isinstance(value, float):
                OGR_F_SetFieldDouble(cogr_feature, i, value)
            elif (isinstance(value, string_types)
            and schema_type in ['date', 'time', 'datetime']):
                if schema_type == 'date':
                    y, m, d, hh, mm, ss, ff = parse_date(value)
                elif schema_type == 'time':
                    y, m, d, hh, mm, ss, ff = parse_time(value)
                else:
                    y, m, d, hh, mm, ss, ff = parse_datetime(value)
                OGR_F_SetFieldDateTime(
                    cogr_feature, i, y, m, d, hh, mm, ss, 0)
            elif (isinstance(value, datetime.date)
            and schema_type == 'date'):
                y, m, d = value.year, value.month, value.day
                OGR_F_SetFieldDateTime(
                    cogr_feature, i, y, m, d, 0, 0, 0, 0)
            elif (isinstance(value, datetime.datetime)
            and schema_type == 'datetime'):
                y, m, d = value.year, value.month, value.day
                hh, mm, ss = value.hour, value.minute, value.second
                OGR_F_SetFieldDateTime(
                    cogr_feature, i, y, m, d, hh, mm, ss, 0)
            elif (isinstance(value, datetime.time)
            and schema_type == 'time'):
                hh, mm, ss = value.hour, value.minute, value.second
                OGR_F_SetFieldDateTime(
                    cogr_feature, i, 0, 0, 0, hh, mm, ss, 0)
            elif isinstance(value, bytes) and schema_type == "bytes":
                string_c = value
                OGR_F_SetFieldBinary(cogr_feature, i, len(value),
                    <unsigned char*>string_c)
            elif isinstance(value, string_types):
                value_bytes = strencode(value, encoding)
                string_c = value_bytes
                OGR_F_SetFieldString(cogr_feature, i, string_c)
            elif value is None:
                set_field_null(cogr_feature, i)
            else:
                raise ValueError("Invalid field type %s" % type(value))
        return cogr_feature


cdef _deleteOgrFeature(void *cogr_feature):
    """Delete an OGR feature"""
    if cogr_feature is not NULL:
        OGR_F_Destroy(cogr_feature)
    cogr_feature = NULL