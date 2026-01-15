import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from model_worker import process_single_article  # We only need the logic, or we can inline specific viz logic

class ForecasterVisualizer:
    
    @staticmethod
    def plot_article_forecast(article_id, df_train_raw, df_future_raw, cfg, train_end_date=None):
        print(f"\n--- Visualizing Article: {article_id} ---")
        
        # 1. Filter Data
        try:
            df_train_sub = df_train_raw.groupby('articleId').get_group(article_id).copy()
            df_future_sub = df_future_raw.groupby('articleId').get_group(article_id).copy()
        except KeyError:
            print(f"Error: Article {article_id} not found in data.")
            return

        # 2. Apply Cutoff if Validation Visualization
        if train_end_date:
            mask_train = df_train_sub['date'] <= pd.to_datetime(train_end_date)
            mask_val = df_train_sub['date'] > pd.to_datetime(train_end_date)
            
            future_actuals = df_train_sub[mask_val].copy()
            df_train_sub = df_train_sub[mask_train].copy()
            
            # Combine future file with validation period
            df_future_sub = pd.concat([future_actuals, df_future_sub], ignore_index=True)

        # 3. Setup (Similar to worker, but we need the model object 'm' for plotting)
        # To avoid code duplication, we repeat the setup steps here for the specific purpose of plotting
        df_train_sub.rename(columns={'date': 'ds', 'sales_volume_index': 'y'}, inplace=True)
        df_future_sub.rename(columns={'date': 'ds'}, inplace=True)

        df_train_sub['is_train'] = True
        df_future_sub['is_train'] = False
        
        df_train_sub['y'] = np.log1p(df_train_sub['y'])

        # Feature Engineering
        cols_needed = ['ds', 'discountPct', 'is_train', 'storeCount', 'FSC_index']
        timeline = pd.concat([df_train_sub[cols_needed], df_future_sub[cols_needed]]).sort_values('ds').reset_index(drop=True)

        regressor_names = ['discountPct', 'storeCount', 'FSC_index']
        
        for lag in cfg.lags:
            col = f'discountPct_lag{lag}'
            timeline[col] = timeline['discountPct'].shift(lag).fillna(0)
            regressor_names.append(col)

        for lead in cfg.leads:
            col = f'discountPct_lead{lead}'
            timeline[col] = timeline['discountPct'].shift(-lead).fillna(0)
            regressor_names.append(col)

        train_features = timeline[timeline['is_train'] == True].drop(columns=['is_train'])
        train_final = pd.merge(df_train_sub[['ds', 'y']], train_features, on='ds', how='left')
        future_features = timeline[timeline['is_train'] == False].drop(columns=['is_train'])
        
        # 5. Fit Model
        m = Prophet(seasonality_mode='additive', yearly_seasonality=True)
        m.add_country_holidays(country_name='NL')
        for reg in regressor_names:
            m.add_regressor(reg)
            
        m.fit(train_final)
        
        # 6. Predict
        forecast = m.predict(future_features)
        
        # 7. Generate Plots
        print("Generating Prophet Plots (Log Scale)...")
        m.plot(forecast)
        plt.title(f"Forecast for Article {article_id} (Log Scale)")
        plt.show()
        
        m.plot_components(forecast)
        plt.show()
        
        # Real Scale Plot
        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(train_final['ds']), np.expm1(train_final['y']), label='Training Data')
        plt.plot(pd.to_datetime(forecast['ds']), np.expm1(forecast['yhat']), label='Forecast', color='orange')
        
        if train_end_date and 'sales_volume_index' in df_future_sub.columns:
             val_data = df_future_sub[df_future_sub['ds'] > pd.to_datetime(train_end_date)]
             if not val_data.empty:
                plt.plot(pd.to_datetime(val_data['ds']), val_data['sales_volume_index'], label='Actual (Validation)', color='green')

        plt.title(f"Real Scale Forecast for Article {article_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()