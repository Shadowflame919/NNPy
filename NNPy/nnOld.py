

import math, pygame, numpy as np
from .other import *

print("Old nn version")

class NN():
	def __init__(self, structure, LEARNING_RATE):

		self.LEARNING_RATE = LEARNING_RATE

		self.MAX_STARTING_WEIGHT = 0.1
		self.MIN_STARTING_WEIGHT = -0.1

		self.structure = structure


		'''
			This stores the weights of the whole network
			Each layer contains neurons which consists of their weights leading to next layer
			Does not need to hold final layer of neurons because they have no weights
		'''
		self.network = []
		for l in range(len(self.structure)-1):	
			layer = []
			# Each layer has as many neurons in the layer, plus one bias neuron
			for n in range(self.structure[l]+1):
				neuron = []
				for w in range(self.structure[l+1]):
					weight = randFloat(self.MIN_STARTING_WEIGHT, self.MAX_STARTING_WEIGHT)
					neuron.append(weight)
				layer.append(neuron)
			self.network.append(layer)


		#print("Network:")
		#for i in self.network:
		#	print("	", len(i))


		''' 
			Used to keep track of derivatives with respect to individual weights
			Knowing dEdW means one can directly apply gradient descent to reduce error (E)
			Allows for training batches as derivatives can be stored (and averaged) across data points
			Has the same data structure as network, each weight needs its own floating point value
			
			To reduce error, set new weight according to:
				new weight = old weight - dEdW*learning rate

		'''
		self.dEdW = []	
		for l in range(len(self.structure)-1):	
			layer = []
			# Each layer has as many neurons in the layer, plus one bias neuron
			for n in range(self.structure[l]+1):
				neuron = []
				for w in range(self.structure[l+1]):
					neuron.append(0)
				layer.append(neuron)
			self.dEdW.append(layer)


		'''
			Used to store derivative of error with respect to the input of each neuron
			First sweep backwards and calculate dEdI for each neuron
			Every dEdI in one layer is needed to calculate dEdI's for neurons in previous layer
			dEdI in a neuron is then used to calculate dEdW for each weight feeding into that neuron
			First layer is not needed, thus first array in dEdI represents network layer 2 of dEdI's
		'''
		self.dEdI = []	
		for l in range(1, len(self.structure)):		# dEdI required for all except first layer
			layer = []
			for n in range(self.structure[l]):		# dEdI is not needed for bias's
				layer.append(0)
			self.dEdI.append(layer)
		

		'''
			Stores the output of each neuron
			Used for calculating multiple parts of backpropagation process
			dOdI for activation function tanh(I) uses output in (1-tanh^2(I))
			dIdW for finding dEdW uses output (I=ow+ow+ow -> dIdW = o)
			dEdO for output layer using squared difference loss function needs dEdO = 2(t-o)

			Contains every neuron output (including bias)
			Might be quicker to not add an extra 1 for bias neurons, 
			 instead I can just add the bias weight to the weighted sum at the end
		'''
		self.netOutput = []
		for l in range(len(self.structure)):
			layer = []
			for n in range(self.structure[l]):
				layer.append(0)

			# Add bias with output value=1 if not final layer
			if l<len(self.structure)-1:
				layer.append(1)

			self.netOutput.append(layer)



	def __str__(self):
		printString = ""
		printString += "Structure: [" + ",".join([str(i) for i in self.structure]) + "]\n"
		for l,layer in enumerate(self.network):
			printString += "Layer " + str(l) + "\n"
			for n,neuron in enumerate(layer):
				printString += "	Neuron " + str(n) + "\n"
				for w,weight in enumerate(neuron):
					printString += "		" + str(w) + ". " + str(weight) + "\n"

		return printString

	def getOutput(self, input, finalOutput=True):

		# Set first layer of netOutput to the current input
		for i in range(self.structure[0]):
			self.netOutput[0][i] = input[i]

		#print(self.netOutput)

		# Output of all layers needs to be calculated except first 
		for l in range(1, len(self.structure)):
			prevNetworkLayer = self.network[l-1]
			prevNetOutputLayer = self.netOutput[l-1]

			# All neurons in current layer except bias produce an output
			for n in range(self.structure[l]):
				layerOutput = 0

				# Sum up weights*output of all neurons in previous layer including bias
				for w in range(self.structure[l-1]+1):	
					#print(self.network[l-1][w][n], self.netOutput[l-1][w])
					layerOutput += prevNetworkLayer[w][n] * prevNetOutputLayer[w]

				# Apply activation function
				layerOutput = math.tanh(layerOutput)
				#if (l < len(self.structure)-1):
				#	layerOutput = math.tanh(layerOutput)

				self.netOutput[l][n] = layerOutput

		if finalOutput:
			return self.netOutput[-1]
		else:
			return self.netOutput


	def applydEdW(self): 	# Applies gradient descent using the current self.dEdW
		# Applies dEdW by altering each weight through gradient descent
		for l in range(len(self.structure)-1):
			for n in range(self.structure[l]+1):	# bias dEdW's are needed in each layer
				for w in range(self.structure[l+1]):
					self.network[l][n][w] += -self.LEARNING_RATE * self.dEdW[l][n][w];

	def resetdEdW(self):
		# Final layer dEdW's do not exist (no weights in final layer)
		for l in range(len(self.structure)-1):
			# bias dEdW's are needed in each layer
			for n in range(self.structure[l]+1):
				for w in range(self.structure[l+1]):
					self.dEdW[l][n][w] = 0

	def getError(self, output, desired):	# Returns squared difference error of two arrays with equal length
		#print("Error: ", output, desired)
		errorSum = 0
		for i in range(len(output)):
			errorSum += (output[i] - desired[i])**2			
		return errorSum


	def train(self, data):

		#print("Starting a train")

		output = self.getOutput(data[0])
		#print("output", output)

		# Calculate error
		errorSumBefore = self.getError(output, data[1])

		#print(data[1], [round(i,2) for i in output], errorSumBefore)

		#print("netoutput:", self.netOutput)
		#print([len(i) for i in self.netOutput])

		self.findGradDescent(data[1])

		self.applydEdW()

		self.resetdEdW()
	
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

		#print("dEdI", self.dEdI)

		# Finds dEdW based on dEdI and adds to current dEdW
		self.finddEdW();



	def findOutputdEdI(self, desired):
		# Finds dEdI for output layer by comparing to desired output
		# Error is the difference squared -> E=(target - desired)^2

		# For each neuron in the final layer
		for n in range(self.structure[-1]):
			# E = (o-t)^2  ->  dEdO = 2(o-t)
			# O = tanh(I)  ->  dOdI = 1 - o^2
			# dEdI = dEdO * dOdI

			# Derivative of difference squared
			self.dEdI[-1][n] = 2 * (self.netOutput[-1][n] - desired[n]);
			#self.dEdI[-1][n] = 4 * (self.netOutput[-1][n] - desired[n])**3;
			#self.dEdI[-1][n] = 8 * (self.netOutput[-1][n] - desired[n])**7;
			
			# Derivative of activation function
			self.dEdI[-1][n] *= (1 - self.netOutput[-1][n]*self.netOutput[-1][n]);
		
	
	def finddEdI(self): 	# Finds dEdI for each layer excluding first and last
		# Starts from end and works its way backwards
		# Last layer is calculated another way and is assumed to already be set
		# First layer doesn't exist as it is not needed in finding dEdW

		# Start 2nd last layer (2nd last layer in dEdI) and work backwards to 2nd layer (1st layer in dEdI)
		for l in range(len(self.structure)-2, 0, -1):
			currLayerNetwork = self.network[l]
			currLayerdEdI = self.dEdI[l]
			prevLayerdEdI = self.dEdI[l-1]

			#print(prevLayerdEdI)

			for n in range(self.structure[l]):	# For each neuron in layer
				currLayerNeuronNetwork = currLayerNetwork[n]
				dEdI = 0	

				# For dEdI of each neuron in next layer
				for w in range(self.structure[l+1]):
					# Each neuron in next layer has a weighted effect on dEdI based on that neurons dEdI
					dEdI += currLayerNeuronNetwork[w] * currLayerdEdI[w]

				#print("dEdI before activation derivative", dEdI, (1 - self.netOutput[l][n]**2), self.netOutput[l][n], [l,n])

				# Activation function derivative
				dEdI *= (1 - self.netOutput[l][n]**2)

				prevLayerdEdI[n] = dEdI 	# Set new dEdI value

				#print("dEdI for neuron", dEdI)



		#print("Made dEdI")
			

	def finddEdW(self):	# Finds dEdW based on current state of dEdI
		# Calculations are added to each value in this.dEdW rather than set to allow for batch training

		for l in range(len(self.structure)-1):		# No weights in last layer
			layer = self.dEdW[l]
			dEdILayer = self.dEdI[l]
			netOutputLayer = self.netOutput[l]

			# Find dEdW for neuron weights, including those originating from bias
			for n in range(self.structure[l]+1):
				neuron = layer[n]
				netOutputNeuron = netOutputLayer[n]

				for w in range(self.structure[l+1]):
					neuron[w] += netOutputNeuron * dEdILayer[w];
			
		

