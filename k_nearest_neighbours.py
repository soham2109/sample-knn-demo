"""
author = Soham Naha
Usage: streamlit run knn_app.py
If streamlit is not installed use, pip install streamlit
The app opens in the browser at localhost:8501
"""

from collections import Counter
import random

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import matplotlib.colors as mplc

color_list = [k for k,v in mplc.cnames.items()]
plt.style.use("bmh")
NUM_CLUSTERS=2

def normalize(X):
	# min-max normalizer
	return (X - X.min())/(X.max()-X.min())


def calc_distance(x1, x2):
	# euclidean distance measure desc
	return np.sqrt(np.sum(np.square(x1-x2), axis=1))


def find_k_nearest_neightbours(X, y, input_data, k):
	# given the dataset points and the unknown label points
	# and the neighbourhood to consider find the
	# k-nearest neighbours of the unknown point
	distance = calc_distance(X, input_data)
	indices = np.argsort(distance)[:k]
	return X[indices], y[indices]


def plot_data_points(X,y, input_data):
	# plot the data points and the unknown data
	fig, ax = plt.subplots(1,1, constrained_layout=True)

	class1_data, class1_labels = X[y==1], y[y==1]
	class2_data, class2_labels = X[y==0], y[y==0]

	ax.scatter(class1_data[:,0],
				  class1_data[:,1],
				  color="r",
				  label="class1",
				  s=15,
				  alpha=0.7)
	ax.scatter(class2_data[:,0],
				  class2_data[:,1],
				  color="b",
				  label="class2",
				  s=15,
				  alpha=0.7)
	ax.scatter(input_data[0],
				  input_data[1],
				  color="g",
				  label="unknown",
				  alpha=0.7,
		  		  s=25 )

	ax.legend(loc="best")
	ax.set_ylim((-0.1, 1.1))
	ax.set_xlim((-0.1, 1.1))
	ax.set_title("Distribution of samples")
	return  fig


def plot_data_points_with_labels(X, y, X_nearest, y_nearest, input_data, label):
	fig, ax = plt.subplots(1,1, constrained_layout=True)
	# select colors for k neigbours
	colors = random.sample(color_list, len(X_nearest))

	class1_data, class1_labels = X[y==1], y[y==1]
	class2_data, class2_labels = X[y==0], y[y==0]

	ax.scatter(class1_data[:,0],
				  class1_data[:,1],
				  color="r",
				  label="class1",
				  s=15,
				  alpha=0.7)
	ax.scatter(class2_data[:,0],
				  class2_data[:,1],
				  color="b",
				  label="class2",
				  s=15,
				  alpha=0.7)
	ax.scatter(input_data[0],
				  input_data[1],
				  color="g",
				  # label="unknown",
				  alpha=0.7,
		  s=25,)
	plt.text(input_data[0]+0.05, input_data[1]-0.05, "Prediction: {}".format(label))

	i = 0
	for x,y in zip(X_nearest, y_nearest):
		ax.plot([x[0], input_data[0]],
				[x[1], input_data[1]],
				colors[i],
				label="neighbour {}".format(i+1), lw=5)
		i+=1

	ax.legend(loc="best")
	ax.set_ylim((-0.1, 1.1))
	ax.set_xlim((-0.1, 1.1))
	ax.set_title("Nearest Neighbour.")
	return  fig



def get_labels(X, y, input_data, k):
	X_nearest, y_nearest = find_k_nearest_neightbours(X, y, input_data, k)
	label = Counter(y_nearest).most_common()[0][0]
	return label, X_nearest, y_nearest


def app():
	st.title("KNN Visualizer using Streamlit")

	st.subheader("Choose a Point (x,y) for Testing.")
	col1, col2, col3 = st.beta_columns(3)
	with col1:
		x_ = st.slider("Choose input feature 1",
						min_value = 0.0,
						max_value = 1.0,
						key="x")
	with col2:
		y_ = st.slider("Choose input feature 2",
						min_value = 0.0,
						max_value = 1.0,
						key="y")
	with col3:
		k = st.slider("Choose the number of neighbours.",
					  min_value=1,
					  max_value=10,
					  key="k")

	input_data = np.array([x_, y_])
	# generate 2-d synthetic data with 2 cluster centers
	X, y = make_blobs(n_samples = 100,
					  n_features = 2,
					  centers = NUM_CLUSTERS,
					  cluster_std=2.1,
					  random_state=0)
	X = normalize(X)

	st.subheader("Visualize the Dataset and neighbours.")
	col1, col2 = st.beta_columns(2)

	with col1:
		st.pyplot(plot_data_points(X, y, input_data))
	with col2:
		label, X_nearest, y_nearest = get_labels(X, y, input_data, k)
		# print(X_nearest.shape)
		st.pyplot(plot_data_points_with_labels(X, y, X_nearest, y_nearest, input_data, label))


if __name__=="__main__":
	app()
