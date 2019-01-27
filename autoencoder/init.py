
'''

	Compresses and uncompresses an image :)
	

'''



import sys, math, random, pygame, json, numpy as np
pygame.init()

import train_mode, test_mode
from NNPy import main, nn

print("Beginning Program")

def setup(self):
	print("Running Setup")

	self.trainingData = []
	self.trainingDataNum = 0
	self.trainingDataLength = 4200		# Train has 42000 data samples
	
	# Gets image data
	print("Extracting Image Data...")
	for k,item in enumerate(open("train.csv")):
		if k==0: continue
		elif k<=self.trainingDataLength:
			imageData = [int(x) for x in item.split(",")]	# Row of data for that image

			# Extract digit type (not currently used)
			imageType = np.array([-1]*10)	
			imageType[imageData.pop(0)] = 1

			# Normalise image data from 0-255 to -1 to 1
			imageData = np.array([2*(x/255)-1 for x in imageData])

			# Add image to training data
			self.trainingData.append(imageData)

			if (k % round(self.trainingDataLength/10) == 0):
				print(str(round(100*k/self.trainingDataLength)) + "%")

		else: break

	print("Image Data Extracted, loaded " + str(len(self.trainingData)) + " images")

	## Setup nn ##
	self.nn = nn.NN([784,128,64,128,784], 0.001)

	#print(self.nn)




def train(self):

	imageData = self.trainingData[self.trainingDataNum]

	# Trains to recreate input
	trainData = [imageData, imageData]

	results = self.nn.train(trainData)

	self.mode.errorList.append(results)

	self.trainingDataNum += 1
	if self.trainingDataNum == len(self.trainingData):
		self.trainingDataNum = 0





main.Main.setup = setup
main.Main.train = train

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


