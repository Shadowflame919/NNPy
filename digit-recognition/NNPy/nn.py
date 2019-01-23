

import math, pygame, numpy as np

class NN():
	def __init__(self, structure, LEARNING_RATE):

		self.LEARNING_RATE = LEARNING_RATE

		self.MAX_STARTING_WEIGHT = 0.1
		self.MIN_STARTING_WEIGHT = -0.1

		self.structure = structure


		'''
			This stores the weights of the whole network
			Each 'layer' of connections should be thought of as the weights between two layers of neurons, 
			rather than being thought of as the neuron layers themselves
			Layers consist of arrays which each represent the connections to a neuron in the next layer
			For example the first array in layer 1 would contain all the weights connecting to neuron 1 in the next layer
			This array would be ordered such that each connection is between the i'th neuron in layer 1, to the 1st neuron in layer 2
			[0->0, 1->0, 2->0, 3->0]
		'''
		self.network = []
		for l in range(len(self.structure)-1):	
			layer = []

			# Each layer contains arrays which each represent all the connections from layer x, to a single neuron in layer x+1
			# Therefore the number of arrays in a layer is the number of neurons in the next layer
			for n in range(self.structure[l+1]):
				neuron = []		# This array holds all the weights that connect to this neuron from the previous layer

				# The number of weights connecting to a neuron is the number of neurons in the previous layer, plus a bias
				for w in range(self.structure[l]+1):
					weight = randFloat(self.MIN_STARTING_WEIGHT, self.MAX_STARTING_WEIGHT)
					neuron.append(weight)

				layer.append(neuron)


			# Add a special array in layer to create an extra 1 on the output of the matrix multiplication to account for a bias input on the next layer calculation
			# Array looks like [0,0,0,0,0,1] where length is the number of neurons in the previous layer + 1
			# Final layer does not technically need this??? Check if final output removes the one when you remove array this on final layer
			if l < len(self.structure)-2:
				biasArray = [0.0] * (self.structure[l]+1)
				biasArray[-1] = 1.0
				layer.append(biasArray)

			layer = np.array(layer)
			self.network.append(layer)

		#print("Network:")
		#for i in self.network:
		#	print("	", i)


		'''
			Stores the output of each layer after network has been fed inputs and fed forward
			Each layer has an array contains outputs for each neuron with a 1 at the end
			This one ensures the matrix multiplication properly takes into account the bias as we assume the input to the bias weight was always a 1
			Final layer does not have an extra one as there are no bias weights to feed into

			Used for calculating multiple parts of backpropagation process
			dOdI for activation function tanh(I) uses output in (1-tanh^2(I))
			dIdW for finding dEdW uses output (I=ow+ow+ow -> dIdW = o)
			dEdO for output layer using squared difference loss function needs dEdO = 2(t-o)
		'''
		self.output = []
		for l in range(len(self.structure)):
			layer = [0.0] * self.structure[l]
			if l<len(self.structure)-1:		# Don't add extra 1 to final layer output
				layer = layer + [1.0]
			layer = np.array(layer)
			self.output.append(layer)

		#print("output", self.output)


		'''
			dEdI for all neurons in each layer
			dEdI has a value for each neuron in the layer, plus an extra 0 on the end for the matrix multiplication to work
			Final layer does NOT have this extra zero, again due to the specifics of the matrix multiplication involved
			First layer is not needed for backpropogating weights
			Every dEdI in one layer is needed to calculate dEdI's for neurons in previous layer (i.e. back propagating through network)
			dEdI in a neuron is then used to calculate dEdW for each weight feeding into that neuron
		'''
		self.dEdI = []
		for l in range(1,len(self.structure)):
			layer = [0.0] * self.structure[l]
			if l<len(self.structure)-1:		# Don't add extra 0 to final layer output
				layer = layer + [0.0]
			layer = np.array(layer)
			self.dEdI.append(layer)

		#print("dEdI", self.dEdI)


		'''
			dEdW for each weight in network (how much the error changes as you change the value of the weight)
			Follows exactly the same structure as self.network
			Knowing dEdW means one can directly apply gradient descent to reduce error (E)
			Allows for training batches as derivatives can be summed together and then applied all at once

			To decend error, set new weight according to:
				new weight = old weight - dEdW*learning rate
		'''
		self.dEdW = []
		for layer in self.network:
			dEdWLayer = layer.copy()
			dEdWLayer.fill(0.0)
			self.dEdW.append(dEdWLayer)

		#print(self.dEdW)


		# Convert to numpy array (dimentions cannot be changed)
		# This cannot be done! Each layer has a different number of neurons and thus differnet array size!! see -> numpy.array([[1,2],[3]])
		#self.network = np.array(self.network, ndmin=3)



	def __str__(self):
		printString = "\nPrinting Network...\n"
		printString += "Structure: [" + ",".join([str(i) for i in self.structure]) + "]\n"
		printString += "Learning Rate: " + str(self.LEARNING_RATE) + "\n"
		for l,layer in enumerate(self.network):
			printString += "Layer " + str(l) + " to " + str(l+1) + "\n"
			for n,neuron in enumerate(layer):
				printString += "	" + str(neuron) + "\n"
				
		return printString


	def getOutput(self, networkInput, finalOutput=True):

		#print("Shape: ", self.network[0].shape)

		# Append a bias [1] to end of input array and set as the layerOutput for the input layer
		layerOutput = np.append(networkInput, 1)	
		self.output[0] = layerOutput

		for k,i in enumerate(self.network):
			# Find the layerOutput of each next layer
			layerOutput = np.matmul(i, layerOutput)

			layerOutput = np.array([math.tanh(x) for x in layerOutput])

			# Makes sure each output except the last has a 1 at the end to correctly perform matrix multiplication
			if k != len(self.network)-1:
				layerOutput[-1] = 1

			self.output[k+1] = layerOutput

			#print(layerOutput)

		#print("output")
		#for i in self.output:
		#	print([x for x in i])

		return layerOutput


	def getError(self, output, desired):	# Returns squared difference error of two arrays with equal length
		errorSum = 0
		for i in range(len(output)):
			errorSum += (output[i] - desired[i])**2			
		return errorSum


	def train(self, data):

		#print("Starting a train")

		output = self.getOutput(data[0])
		#print("output", output)

		# Calculate error before training
		errorSumBefore = self.getError(output, data[1])

		# Goes through multiple steps to ultimately find dEdW for each weight
		self.findGradDescent(data[1])

		# Uses the dEdW's to decend the error
		self.reduceErrorWithdEdW()
	
		return errorSumBefore



	def findGradDescent(self, desired):
		'''
			Calculates the gradient of descent for a single item and adds to dEdW 
			Requires the entire netOutput for that item and the desired output

			Function finds a local dEdI which is attribute of class to prevent recreating each time
			Maybe make dEdI algorithmically global to function.
			This could potentially provide improvements to learning in general
			Probably also faster since dEdW only needs to be found once per batch instead of for each item.
			Although dEdW is just dEdI * output running through it, so not much computation is saved
		'''

		# Find dEdI for last layer
		self.findOutputdEdI(desired)

		# Find dEdI for each preceding layer starting from the 2nd last and not including the first
		self.finddEdI();

		# Finds dEdW based on dEdI and adds to current dEdW
		self.finddEdW();


		#print("dEdI", self.dEdI)


	def findOutputdEdI(self, desired):
		# Finds dEdI for output layer by comparing to desired output
		# Error is the difference squared -> E=(target - desired)^2
		# E = (o-t)^2  ->  dEdO = 2(o-t)
		# O = tanh(I)  ->  dOdI = 1 - o^2
		# dEdI = dEdO * dOdI

		self.dEdI[-1] = np.array([
			 
			# Derivative of difference squared error with respect to neuron output
			2 * (neuronOutput - desired[i]) 

			# Derivative of activation function with respect to neuron input
			* (1 - neuronOutput**2)

		for i,neuronOutput in enumerate(self.output[-1])])

		#print("dEdI", self.dEdI)
		

		# Compare with alternative of
		# for i in range(len(self.dEdI[-1])):
		#	self.dEdI[-1][i] = the derivative value

	
	def finddEdI(self): 	# Finds dEdI for each layer excluding first and last
		# Starts from end and works its way backwards
		# Last layer is calculated another way and is assumed to already be set
		# First layer doesn't exist as it is not needed in finding dEdW

		# Start 2nd last layer (2nd last layer in dEdI) and work backwards to 2nd layer (1st layer in dEdI)
		for l in range(len(self.structure)-2, 0, -1):

			# Find dEdI for layer based on dEdI of next layer and the weights between them
			np.matmul(self.dEdI[l], self.network[l], self.dEdI[l-1])

			# Multiply by activation derivative
			for k,i in enumerate(self.output[l]):
				self.dEdI[l-1][k] *= (1-i**2)
			
			# Set final value to zero to correct matmul
			self.dEdI[l-1][-1] = 0
			
		#print("final dEdI", self.dEdI)



	def finddEdW(self):	# Finds dEdW based on current state of dEdI

		#print("Finding dEdW")

		for k,layer in enumerate(self.network):
			#print("find weights for layer", k)

			# Uses output feeding into weights, and dEdI of neuron weight feeds into to calculate a dEdW map for every weight
			# Could potentially use self.dEdW[k] += np.multiply(...) to instead add the dEdW's for batch training
			np.multiply(self.output[k], self.dEdI[k][:,np.newaxis], self.dEdW[k])


		#for i in self.dEdW:
		#	print(i)



	def reduceErrorWithdEdW(self): 	# Applies gradient descent using the current self.dEdW
		'''
			Uses dEdW by altering each weight to decend error
			If batch training is to apply, should also use this to reset dEdW back to all zeroes 
			so that finddEdW can accumulate self.dEdW rather that just reset it as it currently does

			Should also reduce changes of weights in layers with lots of weights?
			e.g. 
				In [784-32-10], the dEdW of weights throughout the network are about the same.
				This means that if each weight was altered based on their direct dEdW, 
				the connections between 784 and 32 would reduce error considerably more than
				the weights between 32 and 10 and would result in uneven training?

		'''
		
		#print("Final layer before conversion", self.network[-1])
		for k,layer in enumerate(self.dEdW):

			# Convert each dEdW into a change in the weight which would decend error
			layer *= -self.LEARNING_RATE

			# Apply this change to each weight
			self.network[k] += layer


		#print("Final layer after conversion", self.network[-1])




def randFloat(min, max):
	#return 0.1
	return random.random()*(max-min) + min