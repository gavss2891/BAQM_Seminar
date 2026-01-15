import pandas as pd
import numpy as np
from prophet import Prophet
from pathlib import Path
from dataclasses import dataclass
from joblib import Parallel, delayed
import multiprocessing

# --- 1. Configuration Class ---
@dataclass
class Config:
    base_dir: Path = Path.cwd() / 'data' / 'merged_data'
    output_dir: Path = Path.cwd() / 'data' / 'forecasts'
    
    # Lags/Leads
    lags: tuple = () 
    leads: tuple = ()
    
    @property
    def train_path(self):
        return self.base_dir / 'master_train.csv'
    
    @property
    def forecast_path(self):
        # This file serves as both input features AND the output template
        return self.base_dir / 'master_forecast.csv'

# --- 2. The Forecaster Engine ---
class ProphetForecaster:
    def __init__(self, config: Config):
        self.cfg = config
        self.df_train_raw = None
        self.df_future_raw = None

    def load_data(self):
        """Loads raw CSVs into memory."""
        def _read(path, sep=','):
            if not path.exists():
                print(f"File not found: {path}")
                return pd.DataFrame()
            print(f"Loading {path.name}...")
            # Try reading with default comma, fallback or adjust if your master_forecast is actually semicolon
            # Based on previous context, merged_data usually uses comma, but we can be robust.
            try:
                df = pd.read_csv(path, sep=sep, low_memory=False)
            except:
                df = pd.read_csv(path, sep=';', low_memory=False)
                
            df['date'] = pd.to_datetime(df['date'])
            return df.replace('NaN', np.nan).fillna(0)

        self.df_train_raw = _read(self.cfg.train_path)
        self.df_future_raw = _read(self.cfg.forecast_path)
        print("Data loaded successfully.\n")

    @staticmethod
    def _process_single_article(article_id, df_train_sub, df_future_sub, cfg):
        try:
            # --- A. Setup ---
            train = df_train_sub.copy()
            future = df_future_sub.copy()
            
            if train.empty: return None

            train.rename(columns={'date': 'ds', 'sales_volume_index': 'y'}, inplace=True)
            future.rename(columns={'date': 'ds'}, inplace=True)

            train['is_train'] = True
            future['is_train'] = False

            # Log Transform Target
            train['y'] = np.log1p(train['y'])

            # --- B. Feature Engineering ---
            cols_needed = ['ds', 'discountPct', 'is_train']
            timeline = pd.concat([train[cols_needed], future[cols_needed]]).sort_values('ds').reset_index(drop=True)
            
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

            # --- C. Split ---
            train_features = timeline[timeline['is_train'] == True].drop(columns=['is_train'])
            train_final = pd.merge(train[['ds', 'y']], train_features, on='ds', how='left')

            future_final = timeline[timeline['is_train'] == False].drop(columns=['is_train'])
            
            if len(train_final) < 5 or future_final.empty:
                return None

            # --- D. Modeling ---
            m = Prophet(seasonality_mode='additive', yearly_seasonality=True)
            for reg in regressor_names:
                m.add_regressor(reg)
            
            m.fit(train_final)
            
            # --- E. Prediction ---
            forecast = m.predict(future_final)
            forecast_values = np.expm1(forecast['yhat'].values)
            
            # Return just the keys and prediction
            result_df = pd.DataFrame({
                'articleId': article_id,
                'date': future_final['ds'].values,
                'predicted_sales': forecast_values
            })
            
            return result_df

        except Exception:
            return None

    def run_production_forecast(self, n_jobs=-1):
        unique_articles = self.df_future_raw['articleId'].unique()
        n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        
        print(f"--- Starting Production Forecast ---")
        print(f"Target Articles: {len(unique_articles)} | Cores: {n_jobs}")

        grouped_train = self.df_train_raw.groupby('articleId')
        grouped_future = self.df_future_raw.groupby('articleId')
        
        tasks = []
        for art_id in unique_articles:
            if art_id in grouped_train.groups:
                tasks.append((art_id, grouped_train.get_group(art_id), grouped_future.get_group(art_id), self.cfg))

        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(self._process_single_article)(art_id, t, f, c) 
            for art_id, t, f, c in tasks
        )

        final_dfs = [res for res in results if res is not None]
        
        if not final_dfs:
            return pd.DataFrame()

        return pd.concat(final_dfs, ignore_index=True)

# --- 3. Execution ---
if __name__ == "__main__":
    config = Config()
    forecaster = ProphetForecaster(config)
    
    # 1. Load Data
    forecaster.load_data()
    
    # 2. Run Forecast
    predictions = forecaster.run_production_forecast(n_jobs=-1)
    
    # 3. Load Template (master_forecast.csv)
    # We reload it to ensure we have the pristine structure/columns
    print(f"Loading template: {config.forecast_path.name}")
    # Adjust separator if your master_forecast uses semicolons, otherwise default to comma
    try:
        df_template = pd.read_csv(config.forecast_path, sep=',', low_memory=False)
        if len(df_template.columns) <= 1: # check if parsing failed
             df_template = pd.read_csv(config.forecast_path, sep=';', low_memory=False)
    except:
        df_template = pd.read_csv(config.forecast_path, sep=',', low_memory=False)

    # Standardise dates for merging
    df_template['date_parsed'] = pd.to_datetime(df_template['date'])
    predictions['date'] = pd.to_datetime(predictions['date'])
    
    # 4. Merge Predictions into Template
    # Left join to keep all rows in master_forecast (even those without predictions)
    df_merged = pd.merge(
        df_template, 
        predictions, 
        left_on=['articleId', 'date_parsed'], 
        right_on=['articleId', 'date'], 
        how='left'
    )
    
    # 5. Overwrite/Fill 'sales_volume_index'
    # If sales_volume_index already existed (e.g. as 0 or NaN), we update it.
    # We use fillna(0) for missing predictions.
    df_merged['sales_volume_index'] = df_merged['predicted_sales'].fillna(0)
    
    # 6. Cleanup & Export
    # Drop the temporary merge columns
    cols_to_drop = ['date_parsed', 'predicted_sales']
    # If the merge created a duplicate 'date' column (date_y), drop it too
    if 'date_y' in df_merged.columns:
        cols_to_drop.append('date_y')
        df_merged.rename(columns={'date_x': 'date'}, inplace=True)
    
    final_output = df_merged[['date', 'articleId', 'storeCount', 'FSC_index', 'sales_volume_index', 'promo_id']]
    
    # Ensure Output Directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / 'prophet_predictions.csv'
    
    # Write to CSV
    final_output.to_csv(output_path, sep=';', index=False)
    
    print(f"Saved to: {output_path}")
    print(final_output.head())