

import sys, math, pygame, random
import log, button, graph

class Train_Mode():
	def __init__(self, main):
		self.mode = "train"

		self.main = main
		self.screen = main.screen
		self.nn = main.nn
		#self.batchList = main.batchList

		self.font = pygame.font.SysFont(None, 30)



		self.buttonList = [
			button.Button(self.screen, pygame.Rect(535, 90, 30, 30), ">", 40, self.change_graph_next),
			button.Button(self.screen, pygame.Rect(500, 90, 30, 30), "<", 40, self.change_graph_back),
			button.Button(self.screen, pygame.Rect(1050, 150, 150, 30), "Train 1x", 30, self.train_1x),
			button.Button(self.screen, pygame.Rect(1050, 185, 150, 30), "Train 10x", 30, self.train_10x),
			button.Button(self.screen, pygame.Rect(1050, 220, 150, 30), "Train 100x", 30, self.train_100x),
			button.Button(self.screen, pygame.Rect(1050, 255, 150, 30), "Train 1000x", 30, self.train_1000x),
			button.Button(self.screen, pygame.Rect(1050, 290, 150, 30), "Point Map", 30, self.point_map),
			button.Button(self.screen, pygame.Rect(1050, 325, 150, 30), "Curve!", 30, self.curveMore),
		]

		self.errorList = []
		self.errorImprovementList = []
		self.averageInputList = []

		self.pointMap = []

		self.graphNum = 0
		self.graphRect = pygame.Rect(50,130,960,540)
		self.graphList = [
			graph.Graph(
				self.screen, 
				self.graphRect,
				"Error before training",
				self.errorList
			),
			graph.Graph(
				self.screen, 
				self.graphRect, 
				"Point Map",
				self.pointMap,
				-1,
				1
			)
		]
		'''
			graph.Graph(
				self.screen, 
				self.graphRect, 
				"Average error improvement",
				self.errorImprovementList
			),
			graph.Graph(
				self.screen, 
				self.graphRect, 
				"Average Input",
				self.averageInputList 
			)
		]'''




	def update(self, mouseState):	# Starts the main update loop


		self.graphList[self.graphNum].update(mouseState)



	def render(self, mouseState):

		self.graphList[self.graphNum].render(mouseState)

		#text = self.font.render("Batch Num: " + str(self.main.batchNum), True, (0,0,0))
		#self.screen.blit(text, [1050,300])

		

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
		self.main.train()

	def train_10x(self):
		print("Training 10x")
		
		for i in range(10):
			self.main.train()

	def train_100x(self):
		print("Training 100x")
		
		for i in range(100):
			self.main.train()

		self.point_map()

	def train_1000x(self):
		print("Training 1000x")
		
		for i in range(1000):
			self.main.train()

		self.point_map()


	def point_map(self):

		while len(self.pointMap) > 0:
			self.pointMap.pop()

		pointCount = 100
		for i in range(pointCount):
			a = 2*i/pointCount - 1
			#c = math.sin(3.14*a)
			#trainingData = [[a],[c]]

			out = self.main.nn.getOutput([a])[0]

			self.pointMap.append(out)

	def curveMore(self):
		self.main.curve += 0.1
		print("New Curve: " + str(self.main.curve))