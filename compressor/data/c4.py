from compressor.data.base import CalibDataset

__all__ = ["C4CalibDataset"]

@CalibDataset.register("c4")
class C4CalibDataset(CalibDataset):
    data_name_or_path = "allenai/c4"
    data_config = "en"
    split = "train"
    text_key = "text"