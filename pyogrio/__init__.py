from pyogrio._io import ogr_read


def read(path, columns=None):

    ogr_read(path)


def write(path, meta, data, driver=None):
    raise NotImplementedError("Not built!")
