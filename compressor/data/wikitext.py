from compressor.data.base import CalibDataset

__all__ = ["WikiTextCalibDataset"]

@CalibDataset.register("wikitext")
class WikiTextCalibDataset(CalibDataset):
    data_name_or_path = "wikitext"
    data_config = "wikitext-2-raw-v1"
    split = "test"
    text_key = "text"