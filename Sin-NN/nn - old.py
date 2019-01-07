

import math, pygame
import log, other

class NN():
	def __init__(self, structure):

		self.LEARNING_RATE = 10

		self.MAX_STARTING_WEIGHT = 0.1
		self.MIN_STARTING_WEIGHT = -0.1

		self.structure = structure

		self.weightCount = 0

		# Generate network based on structure
		# Network is a 3D array of layers, neurons and weights respectively
		self.network = []
		for l in range(len(self.structure)):
			layer = []
			neuronCount = self.structure[l]
			weightCount = 0
			if not (l == len(self.structure)-1):	
				# Add bias if NOT final layer
				neuronCount += 1

				# Final layer does NOT contain weight propagating to next layer
				weightCount = self.structure[l+1]
				self.weightCount += weightCount

			for n in range(neuronCount):
				neuron = []
				#neuron_dEdW = []
				for w in range(weightCount):
					newWeight = other.randFloat(self.MIN_STARTING_WEIGHT, self.MAX_STARTING_WEIGHT)
					neuron.append(newWeight)
				layer.append(neuron)
			self.network.append(layer)

		print(self.weightCount)

		self.dEdW = []	# Array used inside train function
		# Final layer dEdW's do not exist (no weights in final layer)
		for l in range(len(self.structure)-1):
			self.dEdW.append([])
			# bias dEdW's are needed in each layer
			for n in range(self.structure[l]+1):
				self.dEdW[-1].append([])
				for w in range(self.structure[l+1]):
					self.dEdW[-1][-1].append(0)

		self.dEdI = []	# Array used inside train function
		# First layer dEdI's are not needed for backprop
		for l in range(1, len(self.structure)):	
			self.dEdI.append([])
			for n in range(self.structure[l]):
				self.dEdI[-1].append(0)


	def __str__(self):
		printString = ""
		for l,layer in enumerate(self.network):
			printString += "Layer " + str(l) + "\n"
			for n,neuron in enumerate(layer):
				printString += "	Neuron " + str(n) + "\n"
				for w,weight in enumerate(neuron):
					printString += "		" + str(w) + ". " + str(weight) + "\n"

		return printString

	def getOutput(self, input, finalOutput=True):
		# netOutput starts of with blank input for first layer, 
		# and output of first layer concatenated with bias
		netOutput = [[[],input+[1]]]

		# Calculate IO for all layers except first and last
		for l in range(1,len(self.structure)-1):

			# Include bias to output
			layerOutput = [[0] * self.structure[l], [0] * self.structure[l] + [1]]
			# For each neuron in current layer (excluding bias)
			for n in range(self.structure[l]):
				# For each neuron in previous layer (including bias)
				for m in range(self.structure[l-1]+1):
					#print("Adding " + str(l-1) + "," + str(m) + " to " + str(l) + "," + str(n))
					layerOutput[0][n] += netOutput[l-1][1][m] * self.network[l-1][m][n]

				# Now apply activation function to current value in layerOutput

				# Currently using tanh
				layerOutput[1][n] = math.tanh(layerOutput[0][n])

				# Currently using softplus function for RELU
				#print("Exp: ", layerOutput[0][n])
				#layerOutput[1][n] = math.log(1 + math.exp(layerOutput[0][n]))

			layerOutput[1] += [1]
			netOutput.append(layerOutput)


		# Now calculate final layer
		layerOutput = [[0] * self.structure[-1], [0] * self.structure[-1]]
		softmaxSum = 0
		for n in range(self.structure[-1]):
			for m in range(self.structure[-2]+1):
				layerOutput[0][n] += netOutput[-1][1][m] * self.network[-2][m][n]
			layerOutput[1][n] = math.exp(layerOutput[0][n])
			softmaxSum += layerOutput[1][n]

		for n in range(self.structure[-1]):
			layerOutput[1][n] /= softmaxSum
		netOutput.append(layerOutput)


		if finalOutput:
			return netOutput[-1][1]
		else:
			return netOutput


	def resetdEdW(self):
		# Final layer dEdW's do not exist (no weights in final layer)
		for l in range(len(self.structure)-1):
			# bias dEdW's are needed in each layer
			for n in range(self.structure[l]+1):
				for w in range(self.structure[l+1]):
					self.dEdW[l][n][w] = 0


	def getError(self, output, desired):
		#print("Error: ", output, desired)
		errorSum = 0
		for i in range(len(output)):
			errorSum += (output[i] - desired[i])**2
		return errorSum


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