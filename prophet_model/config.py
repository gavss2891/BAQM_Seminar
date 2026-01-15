from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """
    Central configuration for file paths and model hyperparameters.
    """
    # Base directory
    base_dir: Path = Path.cwd() / 'data' / 'merged_data'
    output_dir: Path = Path.cwd() / 'data' / 'forecasts'
    
    # Feature Engineering: Define lags/leads
    lags: tuple = () 
    leads: tuple = ()
    
    @property
    def train_path(self):
        return self.base_dir / 'master_train.csv'
    
    @property
    def forecast_path(self):
        return self.base_dir / 'master_forecast.csv'