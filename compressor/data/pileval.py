from compressor.data.base import CalibDataset

__all__ = ["PileValCalibDataset"]

@CalibDataset.register("pileval")
class PileValCalibDataset(CalibDataset):
    data_name_or_path = "mit-han-lab/pile-val-backup"
    data_config = None
    split = "validation"
    text_key = "text"