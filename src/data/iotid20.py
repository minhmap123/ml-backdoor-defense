from __future__ import annotations

from pathlib import Path

from .base import DatasetSchema, NumericIDSDataset


class IoTID20Dataset(NumericIDSDataset):
    schema = DatasetSchema(
        name="IoTID20",
        raw_path=Path("data/0_raw/IoTID20/IoT Network Intrusion Dataset.csv"),
        target_column="Cat",
        drop_columns=("Flow_ID", "Src_IP", "Dst_IP", "Src_Port", "Dst_Port", "Timestamp"),
    )
