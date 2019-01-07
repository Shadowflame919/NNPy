
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



import sys, math, random, pygame
pygame.init()

import main, nn


def setup(self):
	print("Setting up")





	## Setup nn ##
	self.nn = nn.NN([1,16,1], 0.01)

	self.curve = 3.14


def train(self):

	#a = random.random()-0.5
	#b = random.random()-0.5
	#c = a + b
	#trainingData = [[a,b],[c]]

	a = 2*random.random()-1
	c = math.sin(self.curve*a)
	trainingData = [[a],[c]]

	results = self.nn.train(trainingData)

	self.mode.errorList.append(results)

	#self.mode.errorList.append(results["errorSumBefore"])
	#self.mode.errorImprovementList.append(results["errorSumBefore"]-results["errorSumAfter"])
	#self.mode.averageInputList.append(results["averageInput"])





def test(self):
	

	trainingData = [[0.2,0.3],[0.5]]

	results = self.nn.getOutput(trainingData[0])

	self.mode.testOutput = ",".join([str(i) for i in results])






main.Main.setup = setup
main.Main.train = train
main.Main.test = test

engine = main.Main()
while True:
	engine.update()
	engine.render()



