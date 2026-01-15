from config import Config
from orchestrator import ProphetOrchestrator

if __name__ == "__main__":
    # 1. Setup
    config = Config()
    forecaster = ProphetOrchestrator(config)
    forecaster.load_data()
    
    # 2. Validation Run
    TRAIN_END = '2023-11-30'
    VALID_END = '2023-12-31'
    
    loss, val_df = forecaster.run_validation(TRAIN_END, VALID_END)
    
    # 3. Visualization
    sample_article_id = '0112d194d01727f5f5ba3c835c9ef20b76a3432d74f7716822a8c07aac1a9374'
    forecaster.plot_forecast(sample_article_id, train_end_date='2023-11-30')
    
    # 4. Production Run
    # prod_forecast = forecaster.run_production_forecast()