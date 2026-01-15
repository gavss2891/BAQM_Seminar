import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from pathlib import Path
from dataclasses import dataclass
from joblib import Parallel, delayed
import multiprocessing

# --- 1. Configuration Class ---
@dataclass
class Config:
    base_dir: Path = Path.cwd() / 'data' / 'merged_data'
    forecast_horizon: int = 31
    lags: tuple = (1, 2, 3) 
    leads: tuple = (1,)
    
    # Custom Split Dates
    train_cutoff_date: str = '2023-10-31'
    validation_cutoff_date: str = '2023-11-30'

    @property
    def train_path(self):
        return self.base_dir / 'master_train.csv'
    
    @property
    def forecast_path(self):
        return self.base_dir / 'master_forecast.csv'

# --- 2. The Forecaster Engine ---
class ProphetForecaster:
    def __init__(self, config: Config):
        self.cfg = config
        self.df_train_raw = None
        self.df_future_raw = None

    def load_data(self):
        """Loads raw CSVs into memory once."""
        def _read(path):
            if not path.exists():
                return self._generate_dummy_data()
            print(f"Loading {path.name}...")
            df = pd.read_csv(path, low_memory=False)
            df['date'] = pd.to_datetime(df['date'])
            return df.replace('NaN', np.nan).fillna(0)

        self.df_train_raw = _read(self.cfg.train_path)
        self.df_future_raw = _read(self.cfg.forecast_path)
        print("Data loaded successfully.\n")

    def _generate_dummy_data(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = []
        for i in range(10): # Increased dummy data for parallel demo
            art_id = f"article_{i}"
            df = pd.DataFrame({
                'date': dates,
                'articleId': art_id,
                'sales_volume_index': np.random.randint(10, 100, len(dates)),
                'discountPct': np.random.choice([0, 0.1, 0.2], len(dates))
            })
            data.append(df)
        return pd.concat(data)

    @staticmethod
    def calculate_smape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        numerator = np.abs(y_pred - y_true)
        denominator = np.abs(y_pred) + np.abs(y_true)
        with np.errstate(divide='ignore', invalid='ignore'):
            smape = numerator / denominator
            smape[denominator == 0] = 0.0
        return 2 * np.mean(smape)

    @staticmethod
    def _process_single_article(article_id, df_train_sub, df_future_sub, cfg):
        """
        Worker function that runs in a separate process.
        It is static and self-contained to avoid pickling the whole class instance.
        """
        try:
            # --- A. Preprocessing ---
            train = df_train_sub.copy()
            future = df_future_sub.copy()
            
            if train.empty: return None

            train.rename(columns={'date': 'ds', 'sales_volume_index': 'y'}, inplace=True)
            future.rename(columns={'date': 'ds'}, inplace=True)

            # Log Transform
            train['y'] = np.log1p(train['y'])

            # Unified Timeline
            cols = ['ds', 'discountPct']
            timeline = pd.concat([train[cols], future[cols]]).sort_values('ds').drop_duplicates('ds').reset_index(drop=True)
            timeline['is_discounted'] = (timeline['discountPct'] > 0).astype(int)
            regressor_names = ['is_discounted']

            for lag in cfg.lags:
                col_name = f'is_discounted_lag{lag}'
                timeline[col_name] = timeline['is_discounted'].shift(lag).fillna(0)
                regressor_names.append(col_name)

            for lead in cfg.leads:
                col_name = f'is_discounted_lead{lead}'
                timeline[col_name] = timeline['is_discounted'].shift(-lead).fillna(0)
                regressor_names.append(col_name)

            train_final = pd.merge(train[['ds', 'y']], timeline, on='ds', how='left')

            # --- B. Evaluation (Train/Test Split) ---
            train_cutoff = pd.to_datetime(cfg.train_cutoff_date)
            valid_cutoff = pd.to_datetime(cfg.validation_cutoff_date)
            
            df = train_final.sort_values('ds').reset_index(drop=True)
            train_df = df[df['ds'] <= train_cutoff].copy()
            valid_df = df[(df['ds'] > train_cutoff) & (df['ds'] <= valid_cutoff)].copy()
            
            if len(train_df) < 10 or len(valid_df) == 0:
                return None

            # Train
            m = Prophet(seasonality_mode='additive', yearly_seasonality=True)
            for reg in regressor_names:
                m.add_regressor(reg)
            m.fit(train_df)
            
            # Predict
            forecast = m.predict(valid_df.drop(columns='y'))
            
            # Inverse Transform
            y_true = np.expm1(valid_df['y'].values)
            y_pred = np.expm1(forecast['yhat'].values)
            
            # Calculate Metric
            smape = ProphetForecaster.calculate_smape(y_true, y_pred)
            
            # Return detailed results
            result_df = pd.DataFrame({
                'articleId': article_id,
                'date': valid_df['ds'].values,
                'actual_sales': y_true,
                'predicted_sales': y_pred,
                'diff': y_pred - y_true
            })
            
            return (article_id, smape, result_df)

        except Exception as e:
            # Catch errors in worker to prevent crash
            print(f"Error processing {article_id}: {e}")
            return None

    def run_batch_evaluation(self, n_jobs=-1):
        """
        Runs evaluation in parallel.
        n_jobs=-1 uses all available cores.
        """
        unique_articles = self.df_train_raw['articleId'].unique()
        cpu_count = multiprocessing.cpu_count()
        n_jobs = cpu_count if n_jobs == -1 else n_jobs
        
        print(f"--- Starting Parallel Evaluation ---")
        print(f"Articles: {len(unique_articles)} | Cores: {n_jobs}")
        

        # Prepare arguments for each task
        # We group the data by articleId first to avoid passing the giant dataframe to every worker
        grouped_train = self.df_train_raw.groupby('articleId')
        grouped_future = self.df_future_raw.groupby('articleId')
        
        tasks = []
        for art_id in unique_articles:
            # Extract only relevant data for this article
            try:
                t_sub = grouped_train.get_group(art_id)
                f_sub = grouped_future.get_group(art_id) if art_id in grouped_future.groups else pd.DataFrame(columns=['date', 'articleId', 'discountPct'])
                tasks.append((art_id, t_sub, f_sub, self.cfg))
            except KeyError:
                continue

        # Execute Parallel Loop
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(self._process_single_article)(art_id, t, f, c) 
            for art_id, t, f, c in tasks
        )

        # Process Results
        metrics_list = []
        all_preds_list = []

        for res in results:
            if res is None: continue
            art_id, smape, preds_df = res
            metrics_list.append({'articleId': art_id, 'sMAPE': smape})
            all_preds_list.append(preds_df)

        metrics_df = pd.DataFrame(metrics_list)
        predictions_df = pd.concat(all_preds_list, ignore_index=True) if all_preds_list else pd.DataFrame()

        print("\n--- Batch Complete ---")
        if not metrics_df.empty:
            print(f"Global Mean sMAPE: {metrics_df['sMAPE'].mean():.4f}")

        return metrics_df, predictions_df

# --- 3. Execution ---
if __name__ == "__main__":
    config = Config(train_cutoff_date='2024-10-31', validation_cutoff_date='2024-11-30')
    forecaster = ProphetForecaster(config)
    forecaster.load_data()
    
    # Run Parallel
    smape_summary, detailed_forecasts = forecaster.run_batch_evaluation(n_jobs=-1)
    
    print("\n--- Summary ---")
    print(smape_summary.head())