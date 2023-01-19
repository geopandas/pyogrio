import pyarrow as pa


class OGRRecordBatchStreamReader(pa.RecordBatchStreamReader):
    _geo_metadata: dict

    @property
    def geo_metadata(self):
        return self._geo_metadata

    @geo_metadata.setter
    def geo_metadata(self, value):
        self._geo_metadata = value
