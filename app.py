from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

app = Flask(__name__)


def clean_data(data_path, predictor_value):
    """
    Function that cleans dataset
    Imputes missing values and return dataset which is ready for next steps
    Parameters:
    data_path -> path to dataset to analyze
    predictor_value -> name of value we want to predict on
    """

    data = pd.read_csv(data_path)
    # Select columns we want to keep, that is datetime, predictor value and features
    columns_to_keep = ['datetime'] + [col for col in data.columns if col.startswith('latest') or col == predictor_value]
    data = data[columns_to_keep]

    # Calculate missing value ratios of columns 
    total_rows = data.shape[0]
    missing_ratios = []
    for column in data.columns:
        missing_ratio = data[column].isnull().sum() / total_rows
        missing_ratios.append({
            'colname': column,
            'missing_ratio': missing_ratio
        })

    # Drop columns with missing values that exceed 50%, probably not good data if most of it is missing
    columns_to_drop = [row['colname'] for row in missing_ratios if row['missing_ratio'] > 0.5]
    data_cleaned = data.drop(columns=columns_to_drop)
    
    # Prepare data befor imputing missing values
    data_cleaned['datetime'] = pd.to_datetime(data_cleaned['datetime'], errors='coerce')
    data_cleaned['hour'] = data_cleaned['datetime'].dt.hour

    # Impute values based on median hourly rate - seems logical, hourly data for time series should be similar
    numeric_data = data_cleaned.drop(columns=['datetime'])
    for column in numeric_data.columns:
        hourly_median = data_cleaned.groupby('hour')[column].median()
        data_cleaned[column] = data_cleaned.apply(
            lambda row: row[column] if pd.notna(row[column]) else hourly_median[row['hour']],
            axis=1
        )

    return data_cleaned


def split_train_test(data_cleaned, predictor='carbon_intensity_avg'):
    """
    Splits dataset to train and test, test is last 24 hours since that is what we want to predict
    Parameters:
    data_cleaned ->  input dataframe of cleaned data
    predictor -> name of predictor variable
    """
    # Split train and test data, test is latest 24 hours.
    train_data = data_cleaned.iloc[:-24]
    test_data = data_cleaned.iloc[-24:]
    # Select covariates that is all columns starting with latest
    covariates = data_cleaned.filter(regex='^latest.*').columns.tolist()

    # Prepare exogenous training,test data
    train_exog = train_data[covariates].values
    test_exog = test_data[covariates].values
    
    # Prepare training,test data for predictors 
    train_endog = train_data[predictor].values
    test_endog = test_data[predictor].values
    
    return train_exog, test_exog, train_endog, test_endog

def find_best_model(train_endog, train_exog, p_values, d_values, q_values, cv_splits=3):
    """
    Finds best ARIMA model for training dataset based on different values of p,d,q
    default cv_split is three fold cross validation
    Parameters:
    train_endog -> Training endogenous variables, that is predictor
    train_exog -> Training exogenous variables, that is covariate matrix
    p_values -> range of p values for AR term
    q_values -> range of q values for MA term
    d_values -> range of d values for level of differencing
    """
    # Initialize best_score and best order varaibles
    best_score = float('inf')
    best_order = None
    
    # Loop through each value of p,d,q and run ARIMA model on cross-validated data
    for p in p_values:
        for d in d_values:
            for q in q_values:
                # Store order of p,d,q in variable
                order = (p, d, q)
                # intialize mse scores
                mse_scores = []
                
                # SPlit data using TimeSeriesSplit based on cv_splits (three_fold is default)
                tscv = TimeSeriesSplit(n_splits=cv_splits)

                # Fit model on each fold and forecast returning an average rmse 
                for train_index, val_index in tscv.split(train_endog):
                    train_endog_cv, val_endog_cv = train_endog[train_index], train_endog[val_index]
                    train_exog_cv, val_exog_cv = train_exog[train_index], train_exog[val_index]

                    model = ARIMA(train_endog_cv, exog=train_exog_cv, order=order)
                    model_fit = model.fit()

                    forecast = model_fit.forecast(steps=len(val_endog_cv), exog=val_exog_cv)
                    rmse = np.sqrt(mean_squared_error(val_endog_cv, forecast))
                    mse_scores.append(rmse)

                # Store RMSE and best order of p,d,q
                avg_rmse = np.mean(mse_scores)
                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_order = order

    return best_order, best_score

def fit_final_arima_model(train_endog, train_exog, best_order):
    """
    Fits arima model with best order of p,d,q,
    Parameters: 
    train_endog -> Training endogenous variables, that is predictor
    train_exog -> Training exogenous variables, that is covariate matrix
    best_order -> Order of best p,d,q value combination
    """
    # Fit model and return model fit
    final_model = ARIMA(train_endog, exog=train_exog, order=best_order)
    final_model_fit = final_model.fit()
    return final_model_fit

@app.route('/forecast', methods=['POST'])
def forecast():
    data_path = request.json.get('data_path')
    predictor_value = request.json.get('predictor_value')

    # Clean data
    data_cleaned = clean_data(data_path, predictor_value)
    # Split train test
    train_exog, test_exog, train_endog, test_endog = split_train_test(data_cleaned, predictor_value)

    # Initialize p,d,q value range
    p_values = range(0, 1)
    d_values = range(0, 1)
    q_values = range(0, 1)
    # Find best arima model
    best_order, best_score = find_best_model(train_endog, train_exog, p_values, d_values, q_values)
    # Fit final model
    final_model_fit = fit_final_arima_model(train_endog, train_exog, best_order)
    # Forecast last 24 hours
    forecast = final_model_fit.forecast(steps=len(test_exog), exog=test_exog)
    # Return predictions as json
    return jsonify(forecast.tolist())

if __name__ == '__main__':
    app.run(debug=True)
