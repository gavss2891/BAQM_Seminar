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
    # Adjusted base directory to align with common project structures if needed
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
        return self.base_dir / 'master_forecast.csv'

# --- 2. The Forecaster Engine ---
class ProphetForecaster:
    def __init__(self, config: Config):
        self.cfg = config
        self.df_train_raw = None
        self.df_future_raw = None

    def load_data(self):
        """Loads raw CSVs into memory."""
        def _read(path):
            if not path.exists():
                print(f"File not found: {path}")
                return pd.DataFrame()
            print(f"Loading {path.name}...")
            # Ensure proper reading if input files also vary in format
            df = pd.read_csv(path, low_memory=False) 
            df['date'] = pd.to_datetime(df['date'])
            return df.replace('NaN', np.nan).fillna(0)

        self.df_train_raw = _read(self.cfg.train_path)
        self.df_future_raw = _read(self.cfg.forecast_path)
        print("Data loaded successfully.\n")


    @staticmethod
    def calculate_smape(y_true, y_pred):
        """
        Calculates sMAPE based on the provided formula:
        sMAPE = 2/N * sum( |F - A| / (|F| + |A|) )
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        numerator = np.abs(y_pred - y_true)
        denominator = np.abs(y_pred) + np.abs(y_true)
        
        # Handle division by zero where both forecast and actual are 0
        # If denominator is 0, the ratio is defined as 0 for this metric
        ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        
        return 2 * np.mean(ratio)

    @staticmethod
    def _process_single_article(article_id, df_train_sub, df_future_sub, cfg):
        """
        Trains on ALL history provided. Predicts on the future file provided.
        """
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

            # --- B. Feature Engineering (Unified Timeline) ---
            cols_needed = ['ds', 'discountPct', 'is_train','storeCount', 'FSC_index']
            
            timeline = pd.concat([train[cols_needed], future[cols_needed]]).sort_values('ds').reset_index(drop=True)
            
            # Create Base Regressor
            regressor_names = ['discountPct', 'storeCount', 'FSC_index']

            # Create Lags
            for lag in cfg.lags:
                col_name = f'discountPct_lag{lag}'
                timeline[col_name] = timeline['discountPct'].shift(lag).fillna(0)
                regressor_names.append(col_name)

            # Create Leads
            for lead in cfg.leads:
                col_name = f'discountPct_lead{lead}'
                timeline[col_name] = timeline['discountPct'].shift(-lead).fillna(0)
                regressor_names.append(col_name)

            # --- C. Split Back to Train/Future ---
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
            
            # Inverse Transform
            forecast_values = np.expm1(forecast['yhat'].values)

            unique_id = future['ds'].astype(str) + future['articleId'].astype(str)

            # Prepare Result
            result_df = pd.DataFrame({
                'id': unique_id,
                'articleId': article_id,
                'date': future_final['ds'].values,
                'predicted_sales': forecast_values
            })
            
            return result_df

        except Exception as e:
            return None
        

    def run_validation(self, train_end_date, valid_end_date, n_jobs=-1):
        """
        Splits historical data into train/valid sets based on dates,
        trains the model, predicts on validation set, and calculates sMAPE.
        """
        print(f"\n--- Starting Validation ---")
        print(f"Training Data:   <= {train_end_date}")
        print(f"Validation Data: > {train_end_date} and <= {valid_end_date}")

        # 1. Filter Data
        mask_train = self.df_train_raw['date'] <= pd.to_datetime(train_end_date)
        mask_valid = (self.df_train_raw['date'] > pd.to_datetime(train_end_date)) & \
                     (self.df_train_raw['date'] <= pd.to_datetime(valid_end_date))

        df_train_sub = self.df_train_raw[mask_train].copy()
        df_valid_sub = self.df_train_raw[mask_valid].copy()

        if df_train_sub.empty or df_valid_sub.empty:
            print("Error: Train or Validation set is empty. Check your dates.")
            return None

        # 2. Prepare Parallel Tasks
        unique_articles = df_valid_sub['articleId'].unique()
        print(f"Articles to validate: {len(unique_articles)}")

        grouped_train = df_train_sub.groupby('articleId')
        grouped_valid = df_valid_sub.groupby('articleId')

        tasks = []
        for art_id in unique_articles:
            try:
                # We need history for this article to train
                t_sub = grouped_train.get_group(art_id)
                # We need validation targets for this article
                v_sub = grouped_valid.get_group(art_id)
                
                # We pass the validation set (v_sub) as the "future" dataframe
                tasks.append((art_id, t_sub, v_sub, self.cfg))
            except KeyError:
                # If an article is in validation but has no history, we skip it
                continue

        # 3. Execute Forecasts
        cpu_count = multiprocessing.cpu_count()
        n_jobs = cpu_count if n_jobs == -1 else n_jobs
        
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(self._process_single_article)(art_id, t, f, c) 
            for art_id, t, f, c in tasks
        )

        final_dfs = [res for res in results if res is not None]

        if not final_dfs:
            print("No predictions generated during validation.")
            return None

        # 4. Consolidate and Score
        df_preds = pd.concat(final_dfs, ignore_index=True)
        
        # Merge predictions with actuals from the validation split
        # We merge on articleId and date to ensure alignment
        cols_actuals = ['articleId', 'date', 'sales_volume_index']
        df_merged = pd.merge(df_valid_sub[cols_actuals], df_preds, on=['articleId', 'date'], how='inner')

        smape_score = self.calculate_smape(df_merged['sales_volume_index'], df_merged['predicted_sales'])
        
        print(f"Validation Complete. Processed {len(df_merged)} observations.")
        print(f"Global sMAPE: {smape_score:.4f}")
        
        return smape_score, df_merged        

    def run_production_forecast(self, n_jobs=-1):
        """
        Runs the forecast generation in parallel.
        """
        unique_articles = self.df_future_raw['articleId'].unique()
        cpu_count = multiprocessing.cpu_count()
        n_jobs = cpu_count if n_jobs == -1 else n_jobs
        
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
            delayed(self._process_single_article)(art_id, t, f, c) 
            for art_id, t, f, c in tasks
        )

        final_dfs = [res for res in results if res is not None]
        
        if not final_dfs:
            print("No forecasts generated.")
            return pd.DataFrame()

        master_forecast = pd.concat(final_dfs, ignore_index=True)
        print(f"\n--- Batch Complete. Generated {len(master_forecast)} rows. ---")
        
        return master_forecast

# --- 3. Execution ---
if __name__ == "__main__":
    config = Config()
    
    forecaster = ProphetForecaster(config)
    forecaster.load_data()
    
    # Run Forecast
    
    TRAIN_END = '2023-11-30'
    VALID_END = '2023-12-31'
    
    loss, val_df = forecaster.run_validation(TRAIN_END, VALID_END)