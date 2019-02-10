
'''

	Compresses and uncompresses an image :)
	

'''

import sys, math, random, pygame, json, numpy as np
import NNPy
import train_mode, test_mode
pygame.init()


print("Beginning Program")


def init(self):
	print("Running Setup")

	# Create image data
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
	self.nn = NNPy.NN([784,128,64,128,784], 0.001)

	#print(self.nn)



	# Create and initialise the modes
	self.modeList = [mode(self) for mode in mainParams["modeList"]]
	self.modeNum = 0
	self.mode = self.modeList[self.modeNum]




def train(self):

	imageData = self.trainingData[self.trainingDataNum]

	# Trains to recreate input
	trainData = [imageData, imageData]

	results = self.nn.train(trainData)

	self.mode.errorList.append(results)

	self.trainingDataNum += 1
	if self.trainingDataNum == len(self.trainingData):
		self.trainingDataNum = 0




NNPy.Main.init = init
NNPy.Main.train = train

mainParams = {
	"modeList": [
		train_mode.Train_Mode,
		test_mode.Test_Mode
	]
}

NNPy.main = NNPy.Main(mainParams)
NNPy.main.init()
while True:
	NNPy.main.update()
	NNPy.main.render()


