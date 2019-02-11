

import sys, math, json, random, pygame, numpy as np
from .button import *
from .nn import *

class Main():
	def __init__(self, params):
		self.res = (1280, 720)
		self.screen = pygame.display.set_mode(self.res)
		self.clock = pygame.time.Clock()

		self.logs = []
		self.font_log = pygame.font.SysFont(None, 24)
		#textrect.centerx = screen.get_rect().centerx

		self.mouseState = [(0,0), False, False, (0,0)]	
		self.buttonList = [
			Button(self.screen, pygame.Rect(1120, 20, 140, 30), "Mode Switch", 28, self.switch_mode)
		]


	def update(self):	# Starts the main update loop

		dt = self.clock.tick(60)/1000

		for event in pygame.event.get():
			if event.type == pygame.QUIT: 
				sys.exit()
			if event.type == pygame.MOUSEBUTTONDOWN:
				self.mouseState[2] = True
				self.mouseState[3] = pygame.mouse.get_pos()
			if event.type == pygame.MOUSEBUTTONUP:
				self.mouseState[1] = True
				self.mouseState[2] = False
				self.mouseState[3] = (0,0)
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_q:
					pass

		self.mouseState[0] = pygame.mouse.get_pos()
		
		self.logs = []
		self.logs.append(round(self.clock.get_fps(), 2))
		self.logs.append("Mode: " + self.mode.mode)
		self.logs.append(self.mouseState[1])


		self.mode.update(self.mouseState, dt)

		# Update main buttons
		for button in self.buttonList:
			button.update(self.mouseState)

		# Update mode buttons
		for button in self.mode.buttonList:
			button.update(self.mouseState)

		# Mouse clicked is only true for one frame
		self.mouseState[1] = False


	def render(self):
		self.screen.fill((230,230,230))

		self.mode.render(self.mouseState)



		self.renderLogs()

		# Render main buttons
		for button in self.buttonList:
			button.render(self.mouseState)

		# Render mode buttons
		for button in self.mode.buttonList:
			button.render(self.mouseState)

		
		pygame.display.flip()


	def renderLogs(self):
		for k,string in enumerate(self.logs):
			text = self.font_log.render(str(string), True, (0,0,0))
			self.screen.blit(text, [10,10+k*20])


	def switch_mode(self):
		self.modeNum += 1
		if self.modeNum == len(self.modeList):
			self.modeNum = 0
		self.mode = self.modeList[self.modeNum]



	def downloadNetwork(self):
		print("Downloading Network")
		fileName = input("File Name: ")
		file = open(fileName, "w")

		fileString = ""
		fileString += json.dumps([int(i) for i in self.nn.structure])
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

		self.nn = NN(netStructure, netLearning)
		self.nn.network = netNetwork

		print("Network uploaded from " + fileName)