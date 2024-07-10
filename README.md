README

HOW TO RUN:

Execute app in terminal with 
 ```
python3 app.py
 ```
Send POST request with curl:
 ```
curl -X POST -H "Content-Type: application/json" -d '{
    "data_path": "mydata.csv",
    "predictor_value": "name_of_predictor_value"
}' http://127.0.0.1:5000/forecast
 ```


## Data Exploration and Cleaning
- Started to look at data see how I could clean it and impute it
- Took a look at numeric and non numeric data
- Saw that it is mostly numeric data with a couple string columns which all have the same value
- Saw there was a lot of missing data, though about some ways to deal with that
- Threw out everything that had more than 50% missing data - figured that columns with majority of missing data would not be useful
- Imputed missing values by median hourly value for rest
- Since it is time series data I assumed that each hour should have some similarity
- Imputing with median hourly value seemed logical to me

## Feature Selection
- Variables starting with latest should be used as features according to description
- Filter out all data except inputted predictor variable and columns that start with latest

## Model Selection
- Thought about what kind of model to fit, I am familiar with ARIMA models so decided to use that and fits well for time series data
- Description asks for forecasts for 24 hours so decided to split the data so test set is last 24 hours
- Training set is all previous data, this is something that I wasn't really sure of, maybe I was supposed to predict on the next unseen 24 hours?
- But didn't really make sense for me since we don't have feature data available for those timeframes
- Thought it would be most straightforward to just leave the last 24 hours as test set and predict on that

## Model Training and Validation
- Model selection was different ARIMA models with p,d,q values
- Use only 0 and 1 values for p,d,q in this solution since using more values would take more time to run
- Even though we would maybe get a better model with other values of p,d,q

## Forecasting
- Select range of p,d,q only form 0 and 1 to save time, would be nice to increase it a little bit to try more models
- Predict on unseen data and return as json

## Afterthougts
- If I had more time I would have done a lot of things a little differently, here are my main thougts:
## Afterthoughts

If I had more time I would have done a lot of things a little differently, here are my main thoughts:

- Spent more time on exploratory data analysis, plot how data looks like to find a better way to impute missing values.
- Probably done some outlier analysis, maybe there are some errors in data, really large values where there shouldn't be.
- Spent more time on finding a better way of splitting data in training and test set, decided on leaving the last 24 hours as test sets since description asked for prediction on next 24 hours. Thinking about it now it was maybe meant to predict on the next unseen 24 hours.
- One mistake I did was using default splits on the cross validated time series data. Thinking about it now it would make more sense to have the validation set to be 24 hours of data like the test set.
- Another thing is handling errors, there are almost no error handling methods in any of the functions. Like if input is wrong or something we should return a error.
- Related to that is creating the functions a little better. The function should be created so that the inputs should have a specific type, same with what the function returns the variables it returns shuold be of specific types like a dataframe or a value.
- Another thing I would have liked to spend more time on is model evaluation. See how well we are predicting on the test set by calculating root squared mean error of test set and real data and see how well it predicts.
- Last thing is packaging the solution so the user doesn't have to have the packages installed. Create a `requirements.txt` file with the listed packages and a `Dockerfile` with the installed packages. That way the user can run it more smoothly.

## Prediction output
|    datetime           |   forecasted_carbon_intensity_avg  |
|-----------------------|-----------------------|
| 2019-09-25 09:00:00+00:00 |         249.232211   |
| 2019-09-25 10:00:00+00:00 |         242.928229   |
| 2019-09-25 11:00:00+00:00 |         241.732926   |
| 2019-09-25 12:00:00+00:00 |         241.179222   |
| 2019-09-25 13:00:00+00:00 |         237.526886   |
| 2019-09-25 14:00:00+00:00 |         269.073713   |
| 2019-09-25 15:00:00+00:00 |         264.069761   |
| 2019-09-25 16:00:00+00:00 |         252.752664   |
| 2019-09-25 17:00:00+00:00 |         257.587545   |
| 2019-09-25 18:00:00+00:00 |         240.653033   |
| 2019-09-25 19:00:00+00:00 |         197.758127   |
| 2019-09-25 20:00:00+00:00 |         185.663735   |
| 2019-09-25 21:00:00+00:00 |         192.682908   |
| 2019-09-25 22:00:00+00:00 |         221.126653   |
| 2019-09-25 23:00:00+00:00 |         226.843506   |
| 2019-09-26 00:00:00+00:00 |         270.093925   |
| 2019-09-26 01:00:00+00:00 |         266.141376   |
| 2019-09-26 02:00:00+00:00 |         277.083526   |
| 2019-09-26 03:00:00+00:00 |         268.697283   |
| 2019-09-26 04:00:00+00:00 |         233.042826   |
| 2019-09-26 05:00:00+00:00 |         258.211519   |
| 2019-09-26 06:00:00+00:00 |         298.617761   |
| 2019-09-26 07:00:00+00:00 |         298.894551   |
| 2019-09-26 08:00:00+00:00 |         286.880666   |

