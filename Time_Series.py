import pandas as pd
from neuralprophet import NeuralProphet #time series neural network using pyTorch
import yfinance as yf #yahoo finance API
from matplotlib import pyplot as plt


msft = yf.Ticker("MSFT")
msft_hist = msft.history(period='max')

msft_hist.drop(columns = ['High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], inplace = True)
msft_hist.reset_index(inplace = True)
msft_hist.rename(columns={'Date': 'ds', 'Open':'y'}, inplace = True)
print(msft_hist.info(verbose = True))
plot = plt.plot(msft_hist['ds'], msft_hist['y'])



model = NeuralProphet() #instantiate the neural network model

metrics = model.fit(msft_hist, freq = 'D', epochs = 500) #train the model
future = model.make_future_dataframe(msft_hist, periods = 30)
forecast = model.predict(future) #return a dataframe with model's 'regression' or estimation of the time series
forecast.head()

plot1 = model.plot(forecast) #plot the model's 'regression' or estimation of the time series
plot2 = model.plot_components(forecast)

plt.show()
