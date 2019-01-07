

import sys, math, pygame, random
from NNPy import button, graph

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
			#button.Button(self.screen, pygame.Rect(720, 100, 30, 30), ">", 40, self.button_0),
			#button.Button(self.screen, pygame.Rect(685, 100, 30, 30), "<", 40, self.button_1),
			#button.Button(self.screen, pygame.Rect(685, 250, 100, 30), "Output", 30, self.getDigitOutput)
		]

		self.digitNum = 0
		self.digitData = None
		self.digitOutput = None


		# used to render
		self.testOutput = ""


	def update(self, mouseState):	# Starts the main update loop

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
