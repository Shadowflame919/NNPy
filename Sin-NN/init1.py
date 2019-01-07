
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



import sys, math, pygame
pygame.init()

import main, nn


def setup(self):
	print("Setting up")


	## Setup training data ##

	self.batchList = []		# Stores a list of batches
	self.batchCount = 50	# Number of batches to store
	self.batchSize = 10		# Number of images per batch
	self.batchNum = 0


	# Gets image data
	for k,item in enumerate(open("train.csv")):
		if k==0: continue
		if k-1<self.batchCount*self.batchSize:
			if (k-1)%self.batchSize==0:		# Create new batch
				self.batchList.append([])
			imageData = [int(x) for x in item.split(",")]
			imageType = [0]*10
			imageType[imageData.pop(0)] = 1

			# Normalise image data from 0-255 to 0-1
			imageData = [(x/255) for x in imageData]

			self.batchList[-1].append([imageData, imageType])
		else:
			break


	## Setup nn ##
	self.nn = nn.NN([784,16,16,10])



def train(self):

	results = self.nn.train(self.batchList[self.batchNum])

	self.mode.errorList.append(results["errorSumBefore"])
	self.mode.errorImprovementList.append(results["errorSumBefore"]-results["errorSumAfter"])
	self.mode.averageInputList.append(results["averageInput"])

	self.batchNum += 1
	if self.batchNum == len(self.batchList):
		self.batchNum = 0



def test(self):
	pass




main.Main.setup = setup
main.Main.train = train
main.Main.test = test

engine = main.Main()
while True:
	engine.update()
	engine.render()



