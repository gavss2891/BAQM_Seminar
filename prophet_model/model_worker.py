import pandas as pd
import numpy as np
from prophet import Prophet

def process_single_article(article_id, df_train_sub, df_future_sub, cfg):
    """
    Worker function: Fits a Prophet model for a single article and generates predictions.
    """
    try:
        # --- A. Setup ---
        train = df_train_sub.copy()
        future = df_future_sub.copy()
        
        if train.empty: return None

        # Prophet requirement: columns must be named 'ds' and 'y'
        train.rename(columns={'date': 'ds', 'sales_volume_index': 'y'}, inplace=True)
        future.rename(columns={'date': 'ds'}, inplace=True)

        train['is_train'] = True
        future['is_train'] = False

        # TRANSFORM: Log-transform target
        train['y'] = np.log1p(train['y'])

        # --- B. Feature Engineering (Unified Timeline) ---
        cols_needed = ['ds', 'discountPct', 'is_train', 'storeCount', 'FSC_index']
        
        timeline = pd.concat([train[cols_needed], future[cols_needed]]).sort_values('ds').reset_index(drop=True)
        
        regressor_names = ['discountPct', 'storeCount', 'FSC_index']

        # Generate Lags
        for lag in cfg.lags:
            col_name = f'discountPct_lag{lag}'
            timeline[col_name] = timeline['discountPct'].shift(lag).fillna(0)
            regressor_names.append(col_name)

        # Generate Leads
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
        m.add_country_holidays(country_name='NL')
        
        for reg in regressor_names:
            m.add_regressor(reg)
        
        m.fit(train_final)
        
        # --- E. Prediction ---
        forecast = m.predict(future_final)
        
        # INVERSE TRANSFORM: expm1
        forecast_values = np.expm1(forecast['yhat'].values)

        unique_id = future['ds'].astype(str) + str(article_id)

        result_df = pd.DataFrame({
            'id': unique_id,
            'articleId': article_id,
            'date': future_final['ds'].values,
            'predicted_sales': forecast_values
        })
        
        return result_df

    except Exception as e:
        return None