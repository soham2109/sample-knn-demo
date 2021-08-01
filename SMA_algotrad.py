import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

@st.cache()
def get_data():
	nflx_ticker = yf.Ticker('NFLX')
	nflx= yf.download('NFLX')
	SMA20 = nflx['Close'].rolling(window = 20).mean()
	SMA50 = nflx['Close'].rolling(window = 50).mean()
	return nflx, SMA20, SMA50

def get_points_above(sma_low, sma_high):
	points_above = {}
	for pair in zip(sma_low, sma_high):
		if pair[0] >= pair[1]:
			date = sma_low[sma_low == pair[0]].index[0]
			points_above[date] = pair[0]

	points_above = pd.Series(points_above, name='Price_Points')
	points_above.index.name = 'Date'

	return points_above
  
def get_crossovers(nflx, SMA20 , SMA50):
	crossovers = pd.DataFrame()
	crossovers['Dates'] = SMA20['Date']
	crossovers['Price'] = [i for i in nflx['Close']]
	crossovers['SMA20'] = SMA20['Close']
	crossovers['SMA50'] = SMA50['Close']
	crossovers['position'] = crossovers['SMA20'] >= crossovers['SMA50']
	crossovers['pre-position'] = crossovers['position'].shift(1)
	crossovers['Crossover'] = np.where(crossovers['position'] == crossovers['pre-position'], False, True)
	crossovers['Crossover'][0] = False
	return crossovers
  
def return_plot(nflx, SMA20, SMA50, crossovers):
	fig, ax = plt.subplots(1,1, constrained_layout=True, figsize=(17,8))
	ax.plot(nflx['Close'][-600:], label='NFLX')
	ax.plot(SMA20[-600:], label='SMA20')
	ax.plot(SMA50[-600:], label='SMA50')
	ax.plot(crossovers.loc[crossovers.Signal == 'Buy']['Dates'][-5:], 
		  crossovers['SMA20'][crossovers.Signal == 'Buy'][-5:],
		  '^', markersize=15, color='g', label='Buy')
	ax.plot(crossovers.loc[crossovers.Signal == 'Sell']['Dates'][-4:], 
		  crossovers['SMA20'][crossovers.Signal == 'Sell'][-4:],
		  'v', markersize=15, color='r', label='Sell')
	ax.legend(loc='upper left', fontsize=15)
	return fig
  

def app():
	nflx, SMA20, SMA50 = get_data()
	points_above_SMA50 = get_points_above(SMA20, SMA50)

	SMA20_reset = SMA20.reset_index()
	SMA50_reset = SMA50.reset_index()

	crossovers = get_crossovers(nflx, SMA20_reset , SMA50_reset)
	crossovers = crossovers.loc[crossovers['Crossover'] == True]
	crossovers = crossovers.reset_index()
	crossovers = crossovers.drop(['position', 'pre-position', 'Crossover', 'index'], axis=1)
	crossovers['Signal'] = np.nan
	crossovers['Binary_Signal'] = 0.0


	for i in range(len(crossovers['SMA20'])):
		if crossovers['SMA20'][i] > crossovers['SMA50'][i]:
			crossovers['Binary_Signal'][i] = 1.0
			crossovers['Signal'][i] = 'Buy'
		else:
			crossovers['Signal'][i] = 'Sell'
	# print(crossovers)
	st.title("Algorithmic Trading using SMA20 and SMA50")
	st.subheader("Netflix Stock Data")
	st.dataframe(nflx.head(10))
	st.subheader("Defining Crossovers after SMA calculations")
	st.dataframe(crossovers.head(10))

	st.subheader("Let's visualize the data obtained.")
	st.pyplot(return_plot(nflx, SMA20, SMA50, crossovers))
  
if __name__=="__main__":
	app()
