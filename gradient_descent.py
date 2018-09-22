'''
Author: Michael Liu
Applied ML Challenge: coding a gradient descent algorithms for linear regression
9/21/2018
'''
from __future__ import division
import math
import numpy as np

def random_generator(seed, length = 1000):
	'''
	this method generates an array of points [(x_1, y_1)...]
	by modifing this method in the future we are able to experience the regression model
	with other statistics
	'''
	np.random.seed(seed)
	a, b = np.random.random((1000, 1)), np.random.random((1000, 1))
	points = list(zip(a,b))
	print('generate random data shape: {}'.format(len(points[0])))
	return points

def loss_function(data_points, m, b):
	'''
	loss = 1/2N * sum((y - (mx + b)) ** 2)
	'''
	x, y = zip(*data_points)
	x, y = np.array(x), np.array(y)
	loss = 1/(len(data_points)) * sum((y - m * x - b) ** 2)
	return loss

def compute_gradient(N, x, y, m, b):
	# for i in range(data_points):
	m_gradient = -1/N * (y - m * x - b) * x
	b_gradient = -1/N * (y - m * x - b)
	return m_gradient, b_gradient

def run_gradient_descent(data_points):
	'''
	pre-define model as linear regression: y = mx + b
	'''
	num_iter = 10000
	learning_rate = 0.1
	m = 0 # initial value for m
	b = 0 # initial value for b
	early_loss = 0 # apply early stopping
	for i in range(num_iter):
		if i % 100 == 0:
			loss = loss_function(data_points, m, b)
			print("loss after {}th 100s of iteration: {}".format(str(i),str(loss)))
			if math.fabs(loss - early_loss) < 1e-10:
				print("early stopping after {}th 100s of iteration".format(str(i)))
				return m, b
			early_loss = loss
		for x, y in data_points:
			m_gradient, b_gradient = compute_gradient(len(data_points), x, y, m, b)
			m -= learning_rate * m_gradient
			b -= learning_rate * b_gradient
	return m, b

if __name__ == "__main__":
	data_points = random_generator(0)
	print("performing gradient descent to optimize linear regression algorithm")
	m, b = run_gradient_descent(data_points)
	print("function is: y = {}x + {}".format(m, b))