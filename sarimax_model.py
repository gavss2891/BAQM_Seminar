import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import itertools
import time
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =====================================================
# LOAD DATA
# =====================================================
merged_data_path = os.path.join(os.getcwd(), 'data', 'merged_data')
master_train_path = os.path.join(merged_data_path, 'master_train.csv')
master_forecast_path = os.path.join(merged_data_path, 'master_forecast.csv')

df_master_train = pd.read_csv(master_train_path, low_memory=False)
df_master_train['date'] = pd.to_datetime(df_master_train['date'])
df_master_train = df_master_train.convert_dtypes()

df_master_forecast = pd.read_csv(master_forecast_path, low_memory=False)
df_master_forecast['date'] = pd.to_datetime(df_master_forecast['date'])
df_master_forecast = df_master_forecast.convert_dtypes()

print("Data loaded successfully!")
print(f"Training shape: {df_master_train.shape}")
print(f"Forecast shape: {df_master_forecast.shape}")

# =====================================================
# FAST AUTO SARIMAX SELECTION
# =====================================================
def auto_arima_selection(endog, exog=None,
                         max_p=2, max_d=1, max_q=2,
                         max_P=1, max_D=1, max_Q=1,
                         s=7, seasonal=True):
    """
    Grid search for best SARIMAX parameters using AIC
    """
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    valid_models = 0
    
    p = range(max_p + 1)
    d = range(max_d + 1)
    q = range(max_q + 1)
    
    if seasonal:
        P = range(max_P + 1)
        D = range(max_D + 1)
        Q = range(max_Q + 1)
        combos = list(itertools.product(p, d, q, P, D, Q))
    else:
        combos = list(itertools.product(p, d, q))
    
    print(f"Testing {len(combos)} SARIMAX configurations...")
    start_time = time.time()
    
    for i, params in enumerate(combos):
        if seasonal:
            order = params[:3]
            seasonal_order = params[3:] + (s,)
        else:
            order = params
            seasonal_order = (0, 0, 0, 0)
        
        # Skip useless model
        if order == (0, 0, 0) and seasonal_order[:3] == (0, 0, 0):
            continue
        
        # Avoid too much differencing
        if order[1] + seasonal_order[1] >= 2:
            continue
        
        try:
            model = SARIMAX(endog, exog=exog,
                           order=order,
                           seasonal_order=seasonal_order,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            fitted = model.fit(
                disp=False,
                maxiter=25,
                method="lbfgs"
            )
            
            if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = order
                best_seasonal_order = seasonal_order
                valid_models += 1
        except:
            continue
        
        if (i+1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Tested {i+1}/{len(combos)} | elapsed {elapsed:.1f}s | valid: {valid_models}")
    
    if best_order is None:
        print("Warning: No valid model found. Using default parameters.")
        best_order = (1, 1, 1)
        best_seasonal_order = (1, 1, 1, s)
    
    return best_order, best_seasonal_order, best_aic

# =====================================================
# LEAD CREATION
# =====================================================
def create_holiday_leads(df_article, lead_days=[1, 2]):
    df = df_article.copy().sort_values('date').reset_index(drop=True)
    for lead in lead_days:
        df[f'holiday_lead_{lead}'] = df['holidayEventIndicator'].shift(-lead).fillna(0).astype(int)
    return df



# =====================================================
# PREPARE DATA FOR ONE ARTICLE
# =====================================================
def prepare_article_data(df_train, article_id, numeric_vars, categorical_vars, indicator_vars, holiday_lead_days):
    """
    Prepare data for a single article (no aggregation)
    Excludes weather variables
    """
    # Filter for this article
    df_article = df_train[df_train['articleId'] == article_id].copy()

    # Apply Leads
    df_article = create_holiday_leads(df_article, holiday_lead_days)
    
    # Prepare target variable
    df_article = df_article.sort_values('date').reset_index(drop=True)
    df_article = df_article.dropna(subset=['sales_volume_index'])
    
    # Handle missing values
    df_article = df_article.dropna(subset=["FSC_index"])
    df_article["discountPct"] = df_article["discountPct"].fillna(0)
    df_article["holidayEventName"] = df_article["holidayEventName"].fillna("__NO_EVENT__")
    
    # Extract target
    y = pd.Series(df_article['sales_volume_index'].values, dtype='float64')
    dates = df_article['date'].reset_index(drop=True)
    
    # Prepare exogenous variables
    exog_list = []
    exog_names = []
    
    # Numeric variables
    for var in numeric_vars:
        if var in df_article.columns:
            values = df_article[var].values.astype('float64')
            exog_list.append(values)
            exog_names.append(var)
    
    # Indicator variables
    for var in indicator_vars:
        if var in df_article.columns:
            values = df_article[var].values.astype('float64')
            exog_list.append(values)
            exog_names.append(var)

    # Lead-day variables
    for day in holiday_lead_days:
        col_name = f'holiday_lead_{day}'
        if col_name in df_article.columns:
            values = df_article[col_name].values.astype('float64')
            exog_list.append(values)
            exog_names.append(col_name)
    
    # Categorical variables - create dummies
    for var in categorical_vars:
        if var in df_article.columns:
            dummies = pd.get_dummies(df_article[var], prefix=var, drop_first=True)
            for col in dummies.columns:
                exog_list.append(dummies[col].values.astype('float64'))
                exog_names.append(col)
    
    # Combine exogenous variables
    exog = None
    if len(exog_list) > 0:
        exog = np.column_stack(exog_list).astype('float64')
        # Scale exogenous variables
        scaler = StandardScaler()
        exog = scaler.fit_transform(exog)
    
    return {
        'y': y,
        'exog': exog,
        'dates': dates,
        'exog_names': exog_names,
        'scaler': scaler if len(exog_list) > 0 else None
    }

# =====================================================
# PREPARE FORECAST EXOGENOUS DATA
# =====================================================
def prepare_forecast_exog(df_forecast, article_id, numeric_vars, categorical_vars, 
                          indicator_vars, scaler, exog_names, holiday_lead_days):
    """
    Prepare exogenous variables from forecast data for the same article
    Aligned exactly with training preparation
    """
    # Filter article
    df_article = df_forecast[df_forecast['articleId'] == article_id].copy()
    df_article = df_article.sort_values('date').reset_index(drop=True)

    # Apply same holiday leads as training
    df_article = create_holiday_leads(df_article, holiday_lead_days)

    # Handle missing values
    df_article = df_article.dropna(subset=["FSC_index"])
    df_article["discountPct"] = df_article["discountPct"].fillna(0)
    df_article["holidayEventName"] = df_article["holidayEventName"].fillna("__NO_EVENT__")

    exog_list = []

    # Numeric variables
    for var in numeric_vars:
        values = df_article[var].values.astype('float64')
        exog_list.append(values)

    # Indicator variables
    for var in indicator_vars:
        values = df_article[var].values.astype('float64')
        exog_list.append(values)

    # Holiday lead variables
    for day in holiday_lead_days:
        col_name = f'holiday_lead_{day}'
        values = df_article[col_name].values.astype('float64')
        exog_list.append(values)

    # Categorical dummies
    dummies_all = []
    for var in categorical_vars:
        dummies = pd.get_dummies(df_article[var], prefix=var, drop_first=True)
        dummies_all.append(dummies)

    if dummies_all:
        dummies_concat = pd.concat(dummies_all, axis=1)
    else:
        dummies_concat = pd.DataFrame()

    # Align dummy columns to training exog_names
    for col in exog_names:
        if col in dummies_concat.columns:
            exog_list.append(dummies_concat[col].values.astype('float64'))
        # If training had a dummy that forecast doesn't, fill with zeros
        elif any(col.startswith(f"{v}_") for v in categorical_vars):
            exog_list.append(np.zeros(len(df_article), dtype='float64'))

    # Combine
    exog_forecast = np.column_stack(exog_list).astype('float64')

    # Scale using training scaler
    if scaler is not None:
        exog_forecast = scaler.transform(exog_forecast)

    return exog_forecast, df_article['date'].values


# =====================================================
# FIT MODEL
# =====================================================
def fit_sarimax_model(article_data):
    """
    Fit SARIMAX model for one article
    """
    y = article_data['y']
    exog = article_data['exog']
    dates = article_data['dates']
    
    print(f"\nObservations: {len(y)}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    if exog is not None:
        print(f"Exogenous variables: {len(article_data['exog_names'])} features")
    
    print("\nAuto-selecting SARIMAX parameters...")
    order, seasonal_order, best_aic = auto_arima_selection(y, exog)
    
    print("\nBest Model:")
    print(f"  Order: {order}")
    print(f"  Seasonal Order: {seasonal_order}")
    print(f"  AIC: {best_aic:.2f}")
    
    print("\nFitting final model...")
    model = SARIMAX(y, exog=exog,
                   order=order,
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    fitted = model.fit(disp=False, maxiter=50, method="lbfgs")
    
    print(f"\nModel fitted successfully!")
    print(f"  AIC: {fitted.aic:.2f}")
    print(f"  BIC: {fitted.bic:.2f}")
    print(f"  Log Likelihood: {fitted.llf:.2f}")
    
    return {
        'model': fitted,
        'order': order,
        'seasonal_order': seasonal_order,
        'y': y,
        'exog': exog,
        'dates': dates,
        'exog_names': article_data['exog_names'],
        'scaler': article_data['scaler']
    }

# =====================================================
# FORECAST
# =====================================================
def forecast_sarimax(model_result, exog_future):
    """
    Generate forecasts using fitted SARIMAX model
    """
    model = model_result['model']
    steps = len(exog_future) if exog_future is not None else 31
    
    print(f"\nGenerating {steps}-period forecast...")
    forecast = model.get_forecast(steps=steps, exog=exog_future)
    mean = forecast.predicted_mean
    ci = forecast.conf_int()
    
    return mean, ci

# =====================================================
# PLOT FUNCTIONS
# =====================================================
def plot_results(model_result, forecast_mean, forecast_ci, forecast_dates):
    """
    Main plot: time series with forecast
    """
    y = model_result['y']
    dates = model_result['dates']
    fitted = model_result['model'].fittedvalues
    
    plt.figure(figsize=(16, 6))
    plt.plot(dates, y, label='Actual', linewidth=1.5, alpha=0.7)
    plt.plot(dates, fitted, label='Fitted', linewidth=1.5, alpha=0.7)
    plt.plot(forecast_dates, forecast_mean, label='Forecast', linewidth=2, color='red')
    plt.fill_between(forecast_dates,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1],
                     color='red', alpha=0.2, label='95% CI')
    plt.legend()
    plt.title('SARIMAX Forecast - Per Article')
    plt.xlabel('Date')
    plt.ylabel('Sales Volume Index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weekly_pattern(model_result):
    """
    Plot weekly pattern (day of week effect)
    """
    y = model_result['y']
    dates = model_result['dates']
    
    # Extract day of week
    df_temp = pd.DataFrame({
        'date': dates,
        'y': y.values,
        'day_of_week': dates.dt.dayofweek,
        'day_name': dates.dt.day_name()
    })
    
    # Average by day of week
    weekly_pattern = df_temp.groupby(['day_of_week', 'day_name'])['y'].mean().reset_index()
    weekly_pattern = weekly_pattern.sort_values('day_of_week')
    
    plt.figure(figsize=(10, 6))
    plt.bar(weekly_pattern['day_name'], weekly_pattern['y'], color='steelblue', edgecolor='black')
    plt.title('Weekly Pattern - Average Sales by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Sales Volume Index')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return weekly_pattern

def plot_yearly_pattern(model_result):
    """
    Plot yearly pattern (month effect)
    """
    y = model_result['y']
    dates = model_result['dates']
    
    # Extract month
    df_temp = pd.DataFrame({
        'date': dates,
        'y': y.values,
        'month': dates.dt.month,
        'month_name': dates.dt.strftime('%b')
    })
    
    # Average by month
    yearly_pattern = df_temp.groupby(['month', 'month_name'])['y'].mean().reset_index()
    yearly_pattern = yearly_pattern.sort_values('month')
    
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_pattern['month'], yearly_pattern['y'], marker='o', linewidth=2, 
             markersize=8, color='darkgreen')
    plt.xticks(yearly_pattern['month'], yearly_pattern['month_name'], rotation=45)
    plt.title('Yearly Pattern - Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sales Volume Index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return yearly_pattern

def plot_diagnostics(model_result):
    """
    Plot model diagnostics: residuals, ACF, PACF
    """
    model = model_result['model']
    residuals = model.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals over time
    axes[0, 0].plot(model_result['dates'], residuals, linewidth=1)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[0, 1].hist(residuals.dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # ACF of residuals
    plot_acf(residuals.dropna(), lags=40, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF of Residuals')
    
    # PACF of residuals
    plot_pacf(residuals.dropna(), lags=40, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title('PACF of Residuals')
    
    plt.tight_layout()
    plt.show()

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    # Define variables (NO weather data)
    numeric_vars = [
        'FSC_index',
        'discountPct'
    ]
    
    indicator_vars = [
        'holidayEventIndicator',
        'workingDayIndicator'
    ]

    holiday_lead_days = [1,2]

    categorical_vars = [
        'category',
        'mainProductGroup',
        'holidayEventName',
        'doWName'
    ]
    
    # Select one article for trial
    article_ids = df_master_train['articleId'].unique()
    if len(article_ids) == 0:
        raise ValueError("No articles found in training data!")
    
    # Change this for trial
    trial_article_id = article_ids[50]
    print("=" * 60)
    print(f"RUNNING SARIMAX FOR ONE ARTICLE (TRIAL)")
    print("=" * 60)
    print(f"Article ID: {trial_article_id[:50]}...")
    
    # Prepare training data for this article
    print("\nPreparing training data...")
    try:
        article_data = prepare_article_data(
            df_master_train,
            trial_article_id,
            numeric_vars,
            categorical_vars,
            indicator_vars,
            holiday_lead_days
        )
        print(f"✓ Training data prepared: {len(article_data['y'])} observations")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        raise
    
    # Fit model
    print("\nFitting SARIMAX model...")
    model_result = fit_sarimax_model(article_data)
    
    # Prepare forecast exogenous data from df_master_forecast
    print("\nPreparing forecast exogenous data...")
    try:
        exog_future, forecast_dates = prepare_forecast_exog(
            df_master_forecast,
            trial_article_id,
            numeric_vars,
            categorical_vars,
            indicator_vars,
            article_data['scaler'],
            article_data['exog_names'],
            holiday_lead_days
        )
        forecast_dates = pd.to_datetime(forecast_dates)
        print(f"✓ Forecast data prepared: {len(exog_future)} periods")
    except Exception as e:
        print(f"✗ Error preparing forecast data: {e}")
        print("Using last known exogenous values instead...")
        # Fallback: use last known values
        if model_result['exog'] is not None:
            exog_future = np.repeat(model_result['exog'][-1:], 31, axis=0)
            forecast_dates = pd.date_range(
                start=model_result['dates'].max() + pd.Timedelta(days=1),
                periods=31,
                freq='D'
            )
        else:
            exog_future = None
            forecast_dates = pd.date_range(
                start=model_result['dates'].max() + pd.Timedelta(days=1),
                periods=31,
                freq='D'
            )
    
    # Generate forecast
    print("\nGenerating forecast...")
    forecast_mean, forecast_ci = forecast_sarimax(model_result, exog_future)
    
    # Plot results
    print("\nPlotting results...")
    plot_results(model_result, forecast_mean, forecast_ci, forecast_dates)
    
    print("\nPlotting weekly pattern...")
    weekly_pattern = plot_weekly_pattern(model_result)
    print("\nWeekly pattern summary:")
    print(weekly_pattern)
    
    print("\nPlotting yearly pattern...")
    yearly_pattern = plot_yearly_pattern(model_result)
    print("\nYearly pattern summary:")
    print(yearly_pattern)
    
    print("\nPlotting diagnostics...")
    plot_diagnostics(model_result)
    
    # Print model summary
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(model_result['model'].summary())
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

