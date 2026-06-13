import pandas as pd

from models.AutoEvent.possession import PossessionDetector
from models.AutoEvent.setpiece import SetPieceDetector
from models.AutoEvent.openplay import OpenPlayDetector

class Postprocessor():
    def __init__(self, tracking: pd.DataFrame, pre_smoothed: bool = False) -> None:
        self.tracking = tracking
        self.pre_smoothed = pre_smoothed
        
        self.poss_result: pd.DataFrame
        self.sp_result: pd.DataFrame
        self.open_result: pd.DataFrame
        
    def run_possesion_detector(self) -> pd.DataFrame:
        return PossessionDetector(self.tracking, pre_smoothed=self.pre_smoothed).run()
    
    def run_setpiece_detector(self, poss_result: pd.DataFrame) -> pd.DataFrame:
        return SetPieceDetector(poss_result).run()
    
    def run_openplay_detector(self, sp_result: pd.DataFrame) -> pd.DataFrame:
        return OpenPlayDetector(sp_result).run()
    
    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Stage 1: Possession
        print("Running PossessionDetector...")
        self.poss_result = self.run_possesion_detector()
        print("PossessionDetector done.")
        
        # Stage 2: Set piece
        print("Running SetPieceDetector...")
        self.sp_result = self.run_setpiece_detector(self.poss_result)
        print("SetPieceDetector done.")
        
        # Stage 3: Open play
        print("Running OpenPlayDetector...")
        self.open_result = self.run_openplay_detector(self.sp_result)
        print("OpenPlayDetector done.")
        
        return self.sp_result, self.open_result