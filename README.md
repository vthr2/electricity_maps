README

HOW TO RUN:

Execute app in terminal with 
 ```
python app.py
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


