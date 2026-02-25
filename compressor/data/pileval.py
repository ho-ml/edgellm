from compressor.data.base import CalibDataset, EvalDataset

__all__ = ["PileValCalibDataset", "PileValEvalDataset"]

@CalibDataset.register("pileval")
class PileValCalibDataset(CalibDataset):
    data_name_or_path = "mit-han-lab/pile-val-backup"
    data_config = None
    split = "validation"
    text_key = "text"

@EvalDataset.register("pileval")
class PileValEvalDataset(EvalDataset):
    data_name_or_path = "mit-han-lab/pile-val-backup"
    data_config = None
    split = "validation"
    text_key = "text"
