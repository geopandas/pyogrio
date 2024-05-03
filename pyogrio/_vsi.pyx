from io import BytesIO
from uuid import uuid4

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from pyogrio._ogr cimport *
from pyogrio._ogr import _get_driver_metadata_item


cdef str get_ogr_vsimem_write_path(object path_or_fp, str driver):
    """ Return the original path or a /vsimem/ path

    If passed a io.BytesIO object, this will return a /vsimem/ path that can be
    used to create a new in-memory file with an extension inferred from the driver
    if possible.  Path will be contained in an in-memory directory to contain
    sibling files (though drivers that create sibling files are not supported for
    in-memory files).

    Caller is responsible for deleting the directory via delete_vsimem_file()

    Parameters
    ----------
    path_or_fp : str or io.BytesIO object
    driver : str
    """

    if not isinstance(path_or_fp, BytesIO):
        return path_or_fp

    # Create in-memory directory to contain auxiliary files
    memfilename = uuid4().hex
    VSIMkdir(f"/vsimem/{memfilename}".encode("UTF-8"), 0666)

    # file extension is required for some drivers, set it based on driver metadata
    ext = ""
    recommended_ext = _get_driver_metadata_item(driver, "DMD_EXTENSIONS")
    if recommended_ext is not None:
        ext = "." + recommended_ext.split(" ")[0]

    path = f"/vsimem/{memfilename}/{memfilename}{ext}"

    # check for existing bytes
    if path_or_fp.getbuffer().nbytes > 0:
        raise NotImplementedError("writing to existing in-memory object is not supported")

    return path


cdef str read_buffer_to_vsimem(bytes bytes_buffer):
    """ Wrap the bytes (zero-copy) into an in-memory dataset

    If the first 4 bytes indicate the bytes are a zip file, the returned path
    will be prefixed with /vsizip/ and suffixed with .zip to enable proper
    reading by GDAL.

    Caller is responsible for deleting the in-memory file via delete_vsimem_file().

    Parameters
    ----------
    bytes_buffer : bytes
    """
    cdef int num_bytes = len(bytes_buffer)

    is_zipped = len(bytes_buffer) > 4 and bytes_buffer[:4].startswith(b"PK\x03\x04")
    ext = ".zip" if is_zipped else ""

    path = f"/vsimem/{uuid4().hex}{ext}"

    # Create an in-memory object that references bytes_buffer
    # NOTE: GDAL does not copy the contents of bytes_buffer; it must remain
    # in scope through the duration of using this file
    vsi_handle = VSIFileFromMemBuffer(path.encode("UTF-8"), <unsigned char *>bytes_buffer, num_bytes, 0)

    if vsi_handle == NULL:
        raise OSError("failed to read buffer into in-memory file")

    if VSIFCloseL(vsi_handle) != 0:
        raise OSError("failed to close in-memory file")

    if is_zipped:
        path = f"/vsizip/{path}"

    return path


cdef read_vsimem_to_buffer(str path, object out_buffer):
    """Copy bytes from in-memory file to buffer

    This will automatically unlink the in-memory file pointed to by path; caller
    is still responsible for calling delete_vsimem_file() to cleanup any other
    files contained in the in-memory directory.

    Parameters:
    -----------
    path : str
        path to in-memory file
    buffer : BytesIO object
    """

    cdef unsigned char *vsi_buffer = NULL
    cdef vsi_l_offset vsi_buffer_size = 0

    try:
        # Take ownership of the buffer to avoid a copy; GDAL will automatically
        # unlink the memory file
        vsi_buffer = VSIGetMemFileBuffer(path.encode("UTF-8"), &vsi_buffer_size, 1)
        if vsi_buffer == NULL:
            raise RuntimeError("could not read bytes from in-memory file")

        # write bytes to buffer
        out_buffer.write(<bytes>vsi_buffer[:vsi_buffer_size])
        # rewind to beginning to allow caller to read
        out_buffer.seek(0)

    finally:
        if vsi_buffer != NULL:
            CPLFree(vsi_buffer)


cdef delete_vsimem_file(str path):
    """ Recursively delete in-memory path or directory containing path

    This is used for final cleanup of an in-memory dataset, which may have been
    created within a directory to contain sibling files.

    Additional VSI handlers may be chained to the left of /vsimem/ in path and
    will be ignored.

    Parameters:
    -----------
    path : str
        path to in-memory file
    """

    if "/vsimem/" not in path:
        return

    root = "/vsimem/" + path.split("/vsimem/")[1].split("/")[0]
    VSIRmdirRecursive(root.encode("UTF-8"))
