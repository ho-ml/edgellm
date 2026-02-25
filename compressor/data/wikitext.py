from compressor.data.base import CalibDataset, EvalDataset

__all__ = ["WikiTextCalibDataset", "WikiTextEvalDataset"]

@CalibDataset.register("wikitext")
class WikiTextCalibDataset(CalibDataset):
    data_name_or_path = "wikitext"
    data_config = "wikitext-2-raw-v1"
    split = "train"
    text_key = "text"

@EvalDataset.register("wikitext")
class WikiTextEvalDataset(EvalDataset):
    data_name_or_path = "wikitext"
    data_config = "wikitext-2-raw-v1"
    split = "test"
    text_key = "text"
