"""
author = Soham Naha
Usage: streamlit run knn_app.py
If streamlit is not installed use, pip install streamlit
The app opens in the browser at localhost:8501
"""
import SMA_algotrad
import k_nearest_neighbours
import streamlit as st

PAGES = {"k-Nearest Neighbours": k_nearest_neighbours,
	 "Simple Moving Average": SMA_algotrad}

def app():
	st.sidebar.title("Navigation Bar")
	option = st.sidebar.radio("Go to", list(PAGES))
	page = PAGES[option]
	page.app()

if __name__=="__main__":
	app()
