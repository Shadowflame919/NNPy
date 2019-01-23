

import sys, math, pygame, random
from .button import Button

class Main():
	def __init__(self, params):
		self.res = (1280, 720)
		self.screen = pygame.display.set_mode(self.res)
		self.clock = pygame.time.Clock()

		self.logs = []
		self.font_log = pygame.font.SysFont(None, 24)
		#textrect.centerx = screen.get_rect().centerx

		self.mouseState = [(0,0), False]	
		self.buttonList = [
			Button(self.screen, pygame.Rect(1120, 20, 140, 30), "Mode Switch", 28, self.switch_mode)
		]

		# Runs the setup function created by user
		self.setup()

		# Modes
		self.modeList = [mode(self) for mode in params["modeList"]]
		#	train_mode.Train_Mode(self),
		#	test_mode.Test_Mode(self)
		#]
		self.modeNum = 0
		self.mode = self.modeList[self.modeNum]



	def update(self):	# Starts the main update loop

		dt = self.clock.tick(60)/1000

		for event in pygame.event.get():
			if event.type == pygame.QUIT: 
				sys.exit()
			if event.type == pygame.MOUSEBUTTONUP:
				self.mouseState[1] = True
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_q:
					print(self.nn)

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


	def trainSum(self, n):		# Trains nn to perform sum of two floats
		batch = []
		for i in range(n):
			a = 1+random.random()
			b = 1+random.random()
			c = a+b
			batch.append([[a,b], [c]])
		newError = self.nn.train(batch)[0]
		self.errorList.append(newError)

	def switch_mode(self):
		self.modeNum += 1
		if self.modeNum == len(self.modeList):
			self.modeNum = 0
		self.mode = self.modeList[self.modeNum]