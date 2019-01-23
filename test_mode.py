

import sys, math, pygame, random, json, numpy as np
from NNPy import button, graph, nn

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
			button.Button(self.screen, pygame.Rect(750, 400, 150, 30), "Full Test", 30, self.fullTest),
			button.Button(self.screen, pygame.Rect(750, 440, 150, 30), "Single Test", 30, self.singleTest),
			button.Button(self.screen, pygame.Rect(750, 500, 150, 30), "Download", 30, self.downloadNetwork),
			button.Button(self.screen, pygame.Rect(750, 540, 150, 30), "Upload", 30, self.uploadNetwork),
			button.Button(self.screen, pygame.Rect(750, 600, 200, 30), "Create Submission", 30, self.createSubmission),
			#button.Button(self.screen, pygame.Rect(720, 100, 30, 30), ">", 40, self.button_0),
			#button.Button(self.screen, pygame.Rect(685, 100, 30, 30), "<", 40, self.button_1),
			#button.Button(self.screen, pygame.Rect(685, 250, 100, 30), "Output", 30, self.getDigitOutput)
		]

		self.digitNum = 0
		self.digitData = None
		self.digitOutput = None


		# used to render
		self.testOutput = ""



		self.testData = []


	def update(self, mouseState, dt):	

		pass

		#if self.digitData == None:
		#	batchNum = math.floor(self.digitNum/len(self.batchList[0]))
		#	imageNum = self.digitNum % len(self.batchList[0])
		#	print(self.digitNum, batchNum, imageNum)
		#	self.digitData = self.batchList[batchNum][imageNum]


	def render(self, mouseState):

		text = self.font.render("TESTING MODE :)", True, (0,0,0))
		self.screen.blit(text, [760,105])

		text = self.font.render("Results: " + self.testOutput, True, (0,0,0))
		self.screen.blit(text, [760,155])

		'''
		# Render digit
		self.renderDigit(self.digitData[0], pygame.Rect(100,100,500,500))


		text = self.font.render("Digit No." + str(self.digitNum), True, (0,0,0))
		self.screen.blit(text, [760,105])

		text = self.font.render("Digit Value: " + str(self.digitData[1]), True, (0,0,0))
		self.screen.blit(text, [685,150])

		# Render digit output
		if self.digitOutput != None:
			for k,i in enumerate(self.digitOutput):
				text = self.font.render(str(k) + ": " + str(i), True, (0,0,0))
				self.screen.blit(text, [685,290 + k*25])

			# Render what network thinks
			text = self.font.render(
				str(self.digitOutput.index(max(self.digitOutput))) + " : " + str(round(100*max(self.digitOutput),3)) + "%",
				True, (0,0,0))
			self.screen.blit(text, [800,255])
		'''
		

	def fullTest(self):
		self.main.test();

	def singleTest(self):
		output = self.main.nn.getOutput(self.main.trainingData[0][0])
		print("First digit output", output)


	def renderDigit(self, data, rect):
		pygame.draw.rect(self.screen, (230,230,230), rect)
		pygame.draw.rect(self.screen, (0,0,0), rect, 3)

		pixelRect = pygame.Rect(0, 0, math.ceil(rect.w/28), math.ceil(rect.h/28))
		for x in range(28):
			for y in range(28):
				pixelRect.topleft = (rect.x + math.ceil(x*(rect.w/28)), rect.y + math.ceil(y*(rect.h/28)))
				pixelColour = [int(data[y*28+x]*255)]*3

				pygame.draw.rect(self.screen, pixelColour, pixelRect)

	def getDigitOutput(self):	# Gets the output for the current testing digit
		self.digitOutput = self.nn.getOutput(self.digitData[0])


	def button_0(self):
		self.digitNum += 1
		self.digitData = None
		self.digitOutput = None
		if self.digitNum == sum([len(x) for x in self.batchList]):
			self.digitNum = 0

	def button_1(self):
		if self.digitNum > 0:
			self.digitNum -= 1
			self.digitData = None
			self.digitOutput = None

	def downloadNetwork(self):
		print("Downloading Network")
		fileName = input("File Name: ")
		file = open(fileName, "w")

		fileString = ""
		fileString += json.dumps(self.nn.structure)
		fileString += "\n" + json.dumps(self.nn.LEARNING_RATE)
		fileString += "\n" + json.dumps([i.tolist() for i in self.nn.network])

		file.write(fileString)
		file.close()

		print("Network saved to " + fileName)

	def uploadNetwork(self):
		print("Uploading Network")
		fileName = input("File Name: ")
		
		netStructure = []
		netLearning = 0
		netNetwork = []
		for i,k in enumerate(open(fileName, "r")):
			if i==0:
				netStructure = json.loads(k)
			elif i==1:
				netLearning = json.loads(k)
			elif i==2:
				netNetwork = json.loads(k)
				netNetwork = [np.array(i) for i in netNetwork]

		self.main.nn = nn.NN(netStructure, netLearning)
		self.main.nn.network = netNetwork
		self.nn = self.main.nn

		print("Network uploaded from " + fileName)



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
			results = self.nn.getOutput(self.testData[k])
			answer = results.argmax()

			submissionString += str(k+1) + "," + str(answer) + "\n"

			if k % math.ceil(testDataLength/10) == 0:
				print(str(round(100*k/testDataLength)) + "%")

		submissionFile = open("submission.csv", "w")
		submissionFile.write(submissionString)
		submissionFile.close()

		print("Submission complete")
