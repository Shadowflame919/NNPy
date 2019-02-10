

import sys, math, pygame, random, numpy as np
import NNPy

class Test_Mode():
	def __init__(self, main):
		self.mode = "test"

		self.main = main
		self.screen = main.screen
		self.nn = main.nn
		#self.batchList = main.batchList

		self.font = pygame.font.SysFont(None, 30)
		#textrect = text.get_rect()
		#textrect.centerx = screen.get_rect().centerx
		#textrect.centery = screen.get_rect().centery


		self.buttonList = [
			NNPy.Button(NNPy.main.screen, pygame.Rect(650, 400, 150, 30), "Full Test", 30, self.fullTest),
			NNPy.Button(NNPy.main.screen, pygame.Rect(650, 440, 150, 30), "Single Test", 30, self.singleTest),
			NNPy.Button(NNPy.main.screen, pygame.Rect(650, 500, 150, 30), "Download", 30, self.main.downloadNetwork),
			NNPy.Button(NNPy.main.screen, pygame.Rect(650, 540, 150, 30), "Upload", 30, self.main.uploadNetwork),
			NNPy.Button(NNPy.main.screen, pygame.Rect(650, 600, 200, 30), "Create Submission", 30, self.createSubmission),

			NNPy.Button(NNPy.main.screen, pygame.Rect(100, 610, 30, 30), "<", 40, self.prevImage),
			NNPy.Button(NNPy.main.screen, pygame.Rect(135, 610, 30, 30), ">", 40, self.nextImage),

			#button.Button(self.screen, pygame.Rect(685, 250, 100, 30), "Output", 30, self.getDigitOutput)
		]


		# Used to store test results
		self.testOutput = ""

		self.testData = []

		self.imageRenderer = NNPy.Image_Renderer(self.screen, pygame.Rect(100,100,500,500))
		self.renderImageNum = 0


	def update(self, mouseState, dt):	

		if len(self.imageRenderer.imageData) == 0:
			self.imageRenderer.setImageData(self.main.trainingData[self.renderImageNum][0])

		#if self.digitData == None:
		#	batchNum = math.floor(self.digitNum/len(self.batchList[0]))
		#	imageNum = self.digitNum % len(self.batchList[0])
		#	print(self.digitNum, batchNum, imageNum)
		#	self.digitData = self.batchList[batchNum][imageNum]


	def render(self, mouseState):

		self.imageRenderer.render()

		text = self.font.render("Image Num: " + str(self.renderImageNum), True, (0,0,0))
		NNPy.main.screen.blit(text, [175,615])

		text = self.font.render("TESTING MODE :)", True, (0,0,0))
		NNPy.main.screen.blit(text, [660,105])

		text = self.font.render("Results: " + self.testOutput, True, (0,0,0))
		NNPy.main.screen.blit(text, [660,155])

		text = self.font.render("Training Data Length: " + str(len(NNPy.main.trainingData)), True, (0,0,0))
		NNPy.main.screen.blit(text, [660,205])



	def fullTest(self):
		NNPy.main.test();

	def singleTest(self):
		output = NNPy.main.nn.getOutput(NNPy.main.trainingData[self.renderImageNum][0])
		self.testOutput = ",".join([str(round(i,2)) for i in output])

	def prevImage(self):
		self.renderImageNum = (self.renderImageNum-1) % len(self.main.trainingData)
		self.imageRenderer.setImageData(NNPy.main.trainingData[self.renderImageNum][0])

	def nextImage(self):
		self.renderImageNum = (self.renderImageNum+1) % len(self.main.trainingData)
		self.imageRenderer.setImageData(NNPy.main.trainingData[self.renderImageNum][0])


	def createSubmission(self):
		print("Creating Submission...")

		print("Extracting Test Image Data...")
		testDataLength = 28000
		if (len(self.testData)==0):
			testFile = open("test.csv")
			for k,item in enumerate(testFile):
				if k==0 or k>testDataLength: continue
				else:
					# Get pixel data
					imageData = [int(x) for x in item.split(",")]	# Row of for that image

					# Normalise pixel data from 0-255 to -1 to 1
					imageData = np.array([2*(x/255)-1 for x in imageData])

					self.testData.append(imageData)

					if (k % math.ceil(testDataLength/10) == 0):
						print(str(round(100*k/testDataLength)) + "%")

			print("Image Data Extracted, loaded " + str(len(self.testData)) + " images")
			testFile.close()

		else:
			print("Test data already extracted")


		# Perform test
		print("Performing test")
		submissionString = "ImageId,Label\n"
		for k,img in enumerate(self.testData):
			results = NNPy.main.nn.getOutput(self.testData[k])
			answer = results.argmax()

			submissionString += str(k+1) + "," + str(answer) + "\n"

			if k % math.ceil(testDataLength/10) == 0:
				print(str(round(100*k/testDataLength)) + "%")

		submissionFile = open("submission.csv", "w")
		submissionFile.write(submissionString)
		submissionFile.close()

		print("Submission complete")


