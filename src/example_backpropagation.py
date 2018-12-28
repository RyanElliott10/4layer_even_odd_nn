import numpy as np
import sys

# A 2 - 3 - 4 - 1 Neural Network

np.random.seed(1)

np.set_printoptions(threshold=np.nan)

def read_data(filename):
	with open(filename) as f:
		data = f.read()
		# print(data[:len(data)-1])
		data = data.split()
		X = []
		y = []
		for i, ex in enumerate(data):
			if (i % 2 == 0):	# even/odd num
				X.append(list(ex))
			
			else:				# label
				y.append(int(ex))
	
	for i, el in enumerate(X):
		X[i]= list(map(int, el))

	return np.asarray(X), np.asarray(y)[np.newaxis].T

def _sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def _sigmoid_prime_z(x):
	return np.exp(x) / ((np.exp(x) + 1) ** 2)

def _sigmoid_prime_a(x):
	return x * (1 - x)


class NeuralNetwork:
	def __init__(self, X, y, nnl2, nnl3, nnl4, alpha):
		self.input = X
		self.output = y
		self.num_training_examples = self.input.shape[0]
		
		self.num_neurons_layer1 = self.input.shape[1]
		self.num_neurons_layer2 = nnl2
		self.num_neurons_layer3 = nnl3
		self.num_neurons_layer4 = nnl4
		
		col_ones = np.ones((self.input.shape[0], 1))
		self.a1 = np.hstack((self.input, col_ones))
		# self.a1 = self.input
		# print(self.a1)
		
		self.theta1 = (2 * np.random.rand(self.num_neurons_layer1+1, self.num_neurons_layer2)) - 1
		self.theta2 = (2 * np.random.rand(self.num_neurons_layer2+1, self.num_neurons_layer3)) - 1
		self.theta3 = (2 * np.random.rand(self.num_neurons_layer3+1, self.num_neurons_layer4)) - 1

		# self.theta1 = (2 * np.random.rand(self.num_neurons_layer2, self.num_neurons_layer1+1)) - 1
		# self.theta2 = (2 * np.random.rand(self.num_neurons_layer3, self.num_neurons_layer2+1)) - 1
		# self.theta3 = (2 * np.random.rand(self.num_neurons_layer4, self.num_neurons_layer3+1)) - 1

		self.alpha = alpha

		# print(self.theta3)
		# print(self.theta2)
		# print(self.theta1)

	def train(self, n_epochs):
		for epoch in range(n_epochs):
			self._feed_forward()
			self._backpropagation()
			towrite = "\rProgress: {0:.3f} %\tError: {1:.10f}         ".format(epoch / n_epochs * 100, np.square((self.a4 - self.output).sum()))
			sys.stdout.write(towrite)
			sys.stdout.flush()
		towrite = "\rProgress: 100.000%"
		sys.stdout.write(towrite)
		sys.stdout.flush()

	def _feed_forward(self):
		self.z2 = np.dot(self.a1, self.theta1)
		self.a2 = _sigmoid(self.z2)
		# print("a2:\n{}".format(self.a2))

		col_ones = np.ones((self.a2.shape[0], 1))
		self.a2 = np.hstack((self.a2, col_ones))

		self.z3 = np.dot(self.a2, self.theta2)
		self.a3 = _sigmoid(self.z3)
		# print("a3:\n{}".format(self.a3))

		col_ones = np.ones((self.a3.shape[0], 1))
		self.a3 = np.hstack((self.a3, col_ones))

		self.z4 = np.dot(self.a3, self.theta3)
		self.a4 = _sigmoid(self.z4)
		# print("a4:\n{}".format(self.a4))

	def predict(self, t_input):
		col_ones = np.ones((t_input.shape[0], 1))
		t_input = np.hstack((t_input, col_ones))

		z2 = np.dot(t_input, self.theta1)
		a2 = _sigmoid(z2)

		col_ones = np.ones((a2.shape[0], 1))
		a2 = np.hstack((a2, col_ones))

		z3 = np.dot(a2, self.theta2)
		a3 = _sigmoid(z3)

		col_ones = np.ones((a3.shape[0], 1))
		a3 = np.hstack((a3, col_ones))

		z4 = np.dot(a3, self.theta3)
		return _sigmoid(z4)

	def _backpropagation(self):
		total_error = np.square(self.a4 - self.output).sum()
		# print("Total error of the network:", total_error)

		delta4 = (self.a4 - self.output) * _sigmoid_prime_z(self.z4)
		inter3 = np.dot(self.theta3, delta4.T)
		delta3 = inter3[:inter3.shape[0]-1, :] * _sigmoid_prime_z(self.z3).T
		inter4 = np.dot(self.theta2, delta3)
		delta2 = inter4[:inter4.shape[0]-1, :] * _sigmoid_prime_z(self.z2).T

		adj3 = np.dot(self.a3.T, delta4)
		adj2 = np.dot(self.a2.T, delta3.T)
		adj1 = np.dot(self.a1.T, delta2.T)

		self.theta3 += -self.alpha * adj3
		self.theta2 += -self.alpha * adj2
		self.theta1 += -self.alpha * adj1


if __name__ == "__main__":
	# X = np.array([ [0,0],
	# 			   [0,1],
	# 			   [1,0],
	# 			   [1,1] ])
	# y = np.array([ [0],
	# 			   [1],
	# 			   [1],
	# 			   [0] ])

	X, y = read_data("training_data.txt")

	# print(X)
	# print(y)
				   
	# Dimensions of the input vector should be: num_examples, num_features
	# Dimensions of the output vector should be: (num_examples, num_possibilities)
	# However, since this is a single class classification problem, this will be
	# (num_examples, 1)

	nnl2 = 3
	nnl3 = 4
	nnl4 = 2
	alpha = 0.001
	epochs = 10000
	
	net = NeuralNetwork(X, y, nnl2, nnl3, nnl4, alpha)
	print("Hyperparameters:\nAlpha: {0}\nEpochs: {1}\n".format(alpha, epochs))
	print("Training the network...")
	net.train(epochs)

	# print("\n\nTesting new scenarios...")
	# X, y = read_data("verification_data.txt")
	# print(net.predict(X) - y)
	print()
