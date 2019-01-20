
'''
	Need an api of sorts

	can create new neural network with many attributes

	
	main stores the current NN and training data

	init.py contains functions:
		setup: 
			- runs when program starts
			- runs before any loops but after main has been created
			- used to create network and init training data
		
		train:
			- used for training network for one iteration?
			- this gets run as often as possible in autotrain mode? 
			  perhaps time how long it takes and then run just enough that a frame takes >1/60 second
			  or run it once for a loop, then twice, and slowly increment until frame takes >1/60th second, then alternate between <1/60th and >1/60th

		test:
			- performs a test for the network
			- test can be logged to varible in main


	
	setup, train and test all run from within the main engine
	"modes" are therefore used as simply interfaces for these modes, and should not contain info necessary for their function
	buttons/graphs can be local to modes

	testing variables?...


'''



import sys, math, random, pygame, json
pygame.init()

import train_mode, test_mode
from NNPy import main,nn

print("Beginning Program")

def setup(self):
	print("Running Setup")

	self.trainingData = []
	self.trainingDataNum = 0
	self.trainingDataLength = 42000		# Train has 42000 data samples
	
	# Gets image data
	print("Extracting Image Data...")
	for k,item in enumerate(open("train.csv")):
		if k==0: continue
		elif k<=self.trainingDataLength:
			imageData = [int(x) for x in item.split(",")]	# Row of for that image
			imageType = [-1]*10	
			imageType[imageData.pop(0)] = 1

			# Normalise image data from 0-255 to -1 to 1
			imageData = [2*(x/255)-1 for x in imageData]

			self.trainingData.append([imageData,imageType])

			if (k % round(self.trainingDataLength/10) == 0):
				print(str(round(100*k/self.trainingDataLength)) + "%")

		else:
			break
	print("Image Data Extracted, loaded " + str(len(self.trainingData)) + " images")

	## Setup nn ##
	self.nn = nn.NN([784,64,64,10], 0.001)




def train(self):

	data = self.trainingData[self.trainingDataNum]

	results = self.nn.train(data)

	self.mode.errorList.append(results)

	self.trainingDataNum += 1
	if self.trainingDataNum == len(self.trainingData):
		self.trainingDataNum = 0


	#self.mode.errorList.append(results["errorSumBefore"])
	#self.mode.errorImprovementList.append(results["errorSumBefore"]-results["errorSumAfter"])
	#self.mode.averageInputList.append(results["averageInput"])





def test(self):
	
	#trainingData = self.trainingData[0]
	#results = self.nn.getOutput(trainingData[0])
	#self.mode.testOutput = ",".join([str(round(i,2)) for i in results])

	# Tests from the end of the training data
	testLength = 2000	
	correct = 0
	print("Performing test on " + str(testLength) + " items")
	for i in range(testLength):
		index = len(self.trainingData) - testLength + i
		results = self.nn.getOutput(self.trainingData[index][0])
		if results.index(max(results)) == self.trainingData[index][1].index(1):
			correct += 1

		if i % round(testLength/10) == 0:
			print(str(round(100*i/testLength)) + "%")

	accuracy = str(round(100*correct/testLength, 2))
	print("Test complete, accuracy of " + accuracy + "%")
	self.mode.testOutput = accuracy + "% (" + str(correct) + " / " + str(testLength) + ")"



main.Main.setup = setup
main.Main.train = train
main.Main.test = test

engineParams = {
	"modeList": [
		train_mode.Train_Mode,
		test_mode.Test_Mode
	]
}

engine = main.Main(engineParams)
while True:
	engine.update()
	engine.render()



