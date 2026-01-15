import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

from config import Config
from model_worker import process_single_article
from utils import calculate_smape
from visualizer import ForecasterVisualizer

class ProphetOrchestrator:
    """
    Orchestrates data loading, parallel execution, and evaluation.
    """
    def __init__(self, config: Config):
        self.cfg = config
        self.df_train_raw = None
        self.df_future_raw = None

    def load_data(self):
        def _read(path):
            if not path.exists():
                print(f"[Warning] File not found: {path}")
                return pd.DataFrame()
            print(f"Loading {path.name}...")
            df = pd.read_csv(path, low_memory=False) 
            df['date'] = pd.to_datetime(df['date'])
            return df.replace('NaN', np.nan).fillna(0)

        self.df_train_raw = _read(self.cfg.train_path)
        self.df_future_raw = _read(self.cfg.forecast_path)
        print("Data loaded successfully.\n")

    def run_validation(self, train_end_date, valid_end_date, n_jobs=-1):
        print(f"\n--- Starting Validation ---")
        print(f"Training Data:   <= {train_end_date}")
        print(f"Validation Data: > {train_end_date} and <= {valid_end_date}")

        # Temporal Split
        mask_train = self.df_train_raw['date'] <= pd.to_datetime(train_end_date)
        mask_valid = (self.df_train_raw['date'] > pd.to_datetime(train_end_date)) & \
                     (self.df_train_raw['date'] <= pd.to_datetime(valid_end_date))

        df_train_sub = self.df_train_raw[mask_train].copy()
        df_valid_sub = self.df_train_raw[mask_valid].copy()

        if df_train_sub.empty or df_valid_sub.empty:
            print("Error: Train or Validation set is empty.")
            return None

        # Prepare Tasks
        unique_articles = df_valid_sub['articleId'].unique()
        print(f"Articles to validate: {len(unique_articles)}")

        grouped_train = df_train_sub.groupby('articleId')
        grouped_valid = df_valid_sub.groupby('articleId')

        tasks = []
        for art_id in unique_articles:
            try:
                t_sub = grouped_train.get_group(art_id)
                v_sub = grouped_valid.get_group(art_id)
                # Pass 'v_sub' as future
                tasks.append((art_id, t_sub, v_sub, self.cfg))
            except KeyError:
                continue

        # Execute Parallel
        n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(process_single_article)(art_id, t, f, c) 
            for art_id, t, f, c in tasks
        )

        final_dfs = [res for res in results if res is not None]

        if not final_dfs:
            print("No predictions generated.")
            return None

        # Consolidate
        df_preds = pd.concat(final_dfs, ignore_index=True)
        cols_actuals = ['articleId', 'date', 'sales_volume_index']
        df_merged = pd.merge(df_valid_sub[cols_actuals], df_preds, on=['articleId', 'date'], how='inner')

        # Score
        smape_score = calculate_smape(df_merged['sales_volume_index'], df_merged['predicted_sales'])
        
        print(f"Validation Complete. Global sMAPE: {smape_score:.4f}")
        return smape_score, df_merged

    def run_production_forecast(self, n_jobs=-1):
        unique_articles = self.df_future_raw['articleId'].unique()
        n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        
        print(f"--- Starting Production Forecast ---")
        print(f"Target Articles: {len(unique_articles)} | Cores: {n_jobs}")

        grouped_train = self.df_train_raw.groupby('articleId')
        grouped_future = self.df_future_raw.groupby('articleId')
        
        tasks = []
        for art_id in unique_articles:
            try:
                t_sub = grouped_train.get_group(art_id)
                f_sub = grouped_future.get_group(art_id)
                tasks.append((art_id, t_sub, f_sub, self.cfg))
            except KeyError:
                continue

        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(process_single_article)(art_id, t, f, c) 
            for art_id, t, f, c in tasks
        )

        final_dfs = [res for res in results if res is not None]
        
        if not final_dfs:
            print("No forecasts generated.")
            return pd.DataFrame()

        master_forecast = pd.concat(final_dfs, ignore_index=True)
        print(f"\n--- Batch Complete. Generated {len(master_forecast)} rows. ---")
        return master_forecast

    def plot_forecast(self, article_id, train_end_date=None):
        """Wrapper for the Visualizer module"""
        ForecasterVisualizer.plot_article_forecast(
            article_id, self.df_train_raw, self.df_future_raw, self.cfg, train_end_date
        )