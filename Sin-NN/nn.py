

import math, pygame
import log, other

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
					weight = other.randFloat(self.MIN_STARTING_WEIGHT, self.MAX_STARTING_WEIGHT)
					neuron.append(weight)
				layer.append(neuron)
			self.network.append(layer)


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
					weight = other.randFloat(self.MIN_STARTING_WEIGHT, self.MAX_STARTING_WEIGHT)
					neuron.append(weight)
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
		for i in range(len(input)):
			self.netOutput[0][i] = input[i]

		#print(self.netOutput)

		# Output of all layers needs to be calculated except first 
		for l in range(1, len(self.structure)):

			# All neurons in current layer except bias produce an output
			for n in range(self.structure[l]):
				layerOutput = 0

				# Sum up weights*output of all neurons in previous layer including bias
				for w in range(self.structure[l-1]+1):	
					#print(self.network[l-1][w][n], self.netOutput[l-1][w])
					layerOutput += self.network[l-1][w][n] * self.netOutput[l-1][w]

				# Apply activation function
				layerOutput = math.tanh(layerOutput)
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
			#errorSum += (output[i] - desired[i])**4		
		return errorSum


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
			for n in range(self.structure[l]):	# For each neuron in layer
				self.dEdI[l-1][n] = 0	# Reset value from previous train

				# For dEdI of each neuron in next layer
				for w in range(self.structure[l+1]):
					# Each neuron in next layer has a weighted effect on dEdI based on that neurons dEdI
					self.dEdI[l-1][n] += self.network[l][n][w] * self.dEdI[l][w]

				# Activation function derivative
				self.dEdI[l-1][n] *= (1 - self.netOutput[l][n]*self.netOutput[l][n])
			

	def finddEdW(self):	# Finds dEdW based on current state of dEdI
		# Calculations are added to each value in this.dEdW rather than set to allow for batch training

		for l in range(len(self.structure)-1):		# No weights in last layer

			# Find dEdW for neuron weights, including those originating from bias
			for n in range(self.structure[l]+1):

				for w in range(self.structure[l+1]):
					self.dEdW[l][n][w] += self.netOutput[l][n] * self.dEdI[l][w];
			
		

	def train(self, data):

		output = self.getOutput(data[0])

		# Calculate error
		errorSumBefore = self.getError(output, data[1])

		print(data, output, errorSumBefore)

		self.findGradDescent(data[1])

		self.applydEdW()

		self.resetdEdW()
	
		return errorSumBefore


	'''

	def train(self, batch):	# Trains nn with a list of IOPairs
		print("Training Batch of length " + str(len(batch)))

		outputBefore = None
		errorSumBefore = 0

		averageInput = 0	# Stores average magnitude of inputs to all neurons

		# Firstly reset dEdW to all zeros
		self.resetdEdW()

		# Loop through each batch item and calculate its unique dEdW 
		for k,item in enumerate(batch):
			print("IOPair No." + str(k), item)

			# First get the actual output for this item
			netOutput = self.getOutput(item[0], False)
			errorSumBefore += self.getError(netOutput[-1][1], item[1])
			print(netOutput)
			print(netOutput[-1][1], item[1])
			outputBefore = netOutput[-1][1]
			#self.printNetOutput(netOutput)

			#print("")
			#print( [x[0] for x in netOutput] )
			averageInput += sum([sum([abs(y) for y in x[0]]) for x in netOutput])

			# Then calculate final layer dEdI's
			for n in range(self.structure[-1]):
				# dEdI for final layer = dEdO * dOdI
				self.dEdI[-1][n] = 2 * (netOutput[-1][1][n] - item[1][n])
				self.dEdI[-1][n] *=  netOutput[-1][1][n] * (1 - netOutput[-1][1][n])

			# Now backpropagate other layers (excluding first) to find dEdI's
			for l in range(len(self.structure)-2, 0, -1):
				# Find dEdI for each neuron in this layer
				for n in range(self.structure[l]):
					# Requires dEdI for each neuron in next layer
					self.dEdI[l-1][n] = 0	# Firsly set to zero!!!
					for m in range(self.structure[l+1]):
						# dEdI does not contain first layer so use (l-1) instead of l
						self.dEdI[l-1][n] += self.dEdI[l][m] * self.network[l][n][m]

					self.dEdI[l-1][n] *= 1 / (1 + math.e**(-netOutput[l][0][n]))


			#print("dEdI", self.dEdI)
			
			# Now find dEdW for each weight (no weights in final layer)
			for l in range(len(self.structure)-1):
				# For each neuron in layer + bias
				for n in range(self.structure[l]+1):
					# For each weight connecting to each neuron in next layer
					for w in range(self.structure[l+1]):
						# dEdW = dEdI * dIdW
						self.dEdW[l][n][w] += self.dEdI[l][w] * netOutput[l][1][n]


		# Now apply gradient deceent with dEdW
		for l in range(len(self.structure)-1):
			# For each neuron in layer + bias
			for n in range(self.structure[l]+1):
				# For each weight connecting to each neuron in next layer
				for w in range(self.structure[l+1]):
					#print(l,n,w)
					# Take average dEdW for each batch item and each weight
					self.dEdW[l][n][w] /= len(batch) * self.structure[l+1]
					self.network[l][n][w] += self.dEdW[l][n][w] * -self.LEARNING_RATE;

		#self.printdEdW()

		# Calculate new error
		errorSumAfter = 0
		outputAfter = None
		for k,item in enumerate(batch):
			netOutput = self.getOutput(item[0])
			errorSumAfter += self.getError(netOutput, item[1])
			outputAfter = netOutput

		#print("Output: ", outputBefore, " -> ", outputAfter)
		print("Error: " + str(errorSumBefore) + " -> " + str(errorSumAfter))

		averageInput /= len(batch) * sum(self.structure[1:])

		return {
			"errorSumBefore": errorSumBefore, 
			"errorSumAfter": errorSumAfter,
			"averageInput": averageInput
		}


	def printNetOutput(self, netOutput):
		print("netOutput:")
		for l in range(len(netOutput)):
			print("Layer " + str(l))
			for n in range(len(netOutput[l][1])):
				print("  Neuron " + str(n))
				if n<len(netOutput[l][0]):
					print("    I: ", netOutput[l][0][n])
				print("    O: ", netOutput[l][1][n])


	def printdEdW(self):
		print("dEdW:")
		# Final layer dEdW's do not exist (no weights in final layer)
		for l in range(len(self.structure)-1):
			print("Layer " + str(l))
			# bias dEdW's are needed in each layer
			for n in range(self.structure[l]+1):
				print("  Neuron " + str(n))
				for w in range(self.structure[l+1]):
					print("    " + str(self.dEdW[l][n][w]))

	'''