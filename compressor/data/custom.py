from compressor.data.base import CalibDataset

__all__ = ["CustomCalibDataset"]

@CalibDataset.register("custom")
class CustomCalibDataset(CalibDataset):
    pass