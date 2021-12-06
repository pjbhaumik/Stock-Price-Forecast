import pandas as pd
from neuralprophet import NeuralProphet #time series neural network using pyTorch
import yfinance as yf #yahoo finance API
from matplotlib import pyplot as plt


msft = yf.Ticker("MSFT")
msft_hist = msft.history(period='10y')

msft_hist.drop(columns = ['High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], inplace = True)
msft_hist.reset_index(inplace = True)
msft_hist.rename(columns={'Date': 'ds', 'Open':'y'}, inplace = True)
print(msft_hist.info(verbose = True))
plot = plt.plot(msft_hist['ds'], msft_hist['y'])


#https://neuralprophet.com/html/hyperparameter-selection.html reference for hyperparameter tuning
model = NeuralProphet(n_forecasts=30, #steps model must predict into the future
    num_hidden_layers = 2, #recommended number of hidden layers 
    d_hidden = 45, #recommended n_lags < d_hidden < n_forecasts where d_hidden = number of units in hidden layer
    n_lags=60, #steps from past to make future predictions
    #n_changepoints=50, #number times trend rate may change
    trend_reg = 0.2,
    yearly_seasonality=12, #number of Fourier terms
    weekly_seasonality= 4, #number of Fourier terms
    daily_seasonality=False,
    batch_size=64,
    epochs=500,
    learning_rate=1.0,
    seasonality_reg = 5,   
    ) #instantiate the neural network model
#help(NeuralProphet)
metrics = model.fit(msft_hist, freq = 'D') #train the model
future = model.make_future_dataframe(msft_hist, periods = 30, n_historic_predictions=120)
forecast = model.predict(future) #return a dataframe with model's 'regression' or estimation of the time series
forecast.head()

plot1 = model.plot(forecast) #plot the model's 'regression' or estimation of the time series
plot2 = model.plot_components(forecast)

plt.show()
