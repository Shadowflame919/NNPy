
import sys, math, pygame, random, numpy as np
import NNPy


class Train_Mode():
	def __init__(self, main):
		self.mode = "train"

		self.font = pygame.font.SysFont(None, 30)

		self.buttonList = [
			NNPy.Button(NNPy.main.screen, pygame.Rect(535, 90, 30, 30), ">", 40, self.change_graph_next),
			NNPy.Button(NNPy.main.screen, pygame.Rect(500, 90, 30, 30), "<", 40, self.change_graph_back),
			NNPy.Button(NNPy.main.screen, pygame.Rect(1050, 150, 150, 30), "Train 1x", 30, self.train_1x),
			NNPy.Button(NNPy.main.screen, pygame.Rect(1050, 185, 150, 30), "Train 10x", 30, self.train_10x),
			NNPy.Button(NNPy.main.screen, pygame.Rect(1050, 220, 150, 30), "Train 100x", 30, self.train_100x),
			NNPy.Button(NNPy.main.screen, pygame.Rect(1050, 255, 150, 30), "Train: False", 30, self.toggleAutoTrain),
		]

		self.errorList = []
		self.errorImprovementList = []
		self.averageInputList = []

		self.pointMap = []

		self.graphNum = 0
		self.graphRect = pygame.Rect(50,130,960,540)
		self.graphList = [
			NNPy.Graph(
				NNPy.main.screen, 
				self.graphRect,
				"Error before training",
				self.errorList
			)
		]

		self.autoTrain = False
		self.autoTrainSpeed = 1		# Number of times to train per update cycle
		self.autoTrainTime = 0		# Time done autotraining


	def update(self, mouseState, dt):	

		if (self.autoTrain):
			for i in range(math.floor(self.autoTrainSpeed)):
				NNPy.main.train()

			# Keeps rendering at atleast 30fps
			if (dt <= 1/30):	
				self.autoTrainSpeed += 1
			elif (self.autoTrainSpeed > 1):
				self.autoTrainSpeed -= 1
			
			self.autoTrainTime += dt
			
		self.graphList[self.graphNum].update(mouseState)


	def render(self, mouseState):

		self.graphList[self.graphNum].render(mouseState)

		text = self.font.render("Train Speed: " + str(self.autoTrainSpeed), True, (0,0,0))
		NNPy.main.screen.blit(text, [1050,500])

		text = self.font.render("Time Trained: " + str(round(self.autoTrainTime)), True, (0,0,0))
		NNPy.main.screen.blit(text, [1050,540])

		text = self.font.render(str(NNPy.main.nn.structure), True, (0,0,0))
		NNPy.main.screen.blit(text, [1050,580])

		text = self.font.render(str(NNPy.main.nn.LEARNING_RATE), True, (0,0,0))
		NNPy.main.screen.blit(text, [1050,620])


		

	def change_graph_next(self):
		self.graphNum += 1
		if self.graphNum == len(self.graphList):
			self.graphNum = 0

	def change_graph_back(self):
		self.graphNum -= 1
		if self.graphNum < 0:
			self.graphNum = len(self.graphList) - 1



	def train_1x(self):
		print("Training Once")
		NNPy.main.train()

	def train_10x(self):
		print("Training 10x")
		for i in range(10):
			NNPy.main.train()

	def train_100x(self):
		print("Training 100x")
		for i in range(100):
			NNPy.main.train()

	def toggleAutoTrain(self):
		self.autoTrain = not self.autoTrain
		return "Train: " + str(self.autoTrain)