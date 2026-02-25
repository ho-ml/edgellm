from compressor.data.base import CalibDataset, EvalDataset

__all__ = ["C4CalibDataset", "C4EvalDataset"]

@CalibDataset.register("c4")
class C4CalibDataset(CalibDataset):
    data_name_or_path = "allenai/c4"
    data_config = "en"
    split = "train"
    text_key = "text"

@EvalDataset.register("c4")
class C4EvalDataset(EvalDataset):
    data_name_or_path = "allenai/c4"
    data_config = "en"
    split = "validation"
    text_key = "text"
