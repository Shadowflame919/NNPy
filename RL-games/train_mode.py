
import sys, math, pygame, random, numpy as np
import NNPy, tictactoe


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

		self.randomWinRate = []
		self.randomWinRateFirst = []
		self.randomWinRateSecond = []


		self.graphNum = 0
		self.graphRect = pygame.Rect(50,130,960,540)
		self.graphList = [
			NNPy.Graph(
				NNPy.main.screen, 
				self.graphRect,
				"Best bot VS Random",
				self.randomWinRate
			),
			NNPy.Graph(
				NNPy.main.screen, 
				self.graphRect,
				"Best bot VS Random (Playing first)",
				self.randomWinRateFirst
			),
			NNPy.Graph(
				NNPy.main.screen, 
				self.graphRect,
				"Best bot VS Random (Playing second)",
				self.randomWinRateSecond
			)
		]

		self.autoTrain = False
		self.autoTrainSpeed = 1		# Number of times to train per update cycle
		self.autoTrainTime = 0		# Time done autotraining



		self.botGame = tictactoe.TicTacToeGame()
		self.botScores = np.zeros(NNPy.main.botCount, dtype="int16")






	def train(self):

		'''
		testGames = 1000
		firstWins = 0
		secondWins = 0
		ties = 0
		for i in range(testGames):
			randomWinnerFirst = self.botGame.playGameAgainstBots(self.botGame.botGood, self.botGame.botRandom)
			if randomWinnerFirst == 0:
				firstWins += 1
			elif randomWinnerFirst == -1:
				ties += 1

			randomWinnerSecond =  self.botGame.playGameAgainstBots(self.botGame.botRandom, self.botGame.botGood)
			if randomWinnerSecond == 1:
				secondWins += 1
			elif randomWinnerSecond == -1:
				ties += 1

		print(firstWins, secondWins, ties)

		return'''

		'''
			A single train runs a whole generation? game?

			Bots play each other bot in a game


			About 20 games per tick can be played

		'''

		# Play games between each possible pair of bots
		'''
		for a in range(NNPy.main.botCount):
			for b in range(a):
				# Play a whole game when a is player 1, and b is player -1, and then the other way around

				winnerAB = self.botGame.playGame(a,b)
				winnerBA = self.botGame.playGame(b,a)

				#print(a,b,"->",winnerAB)
				#print(b,a,"->",winnerBA)

				if winnerAB != -1:
					self.botScores[winnerAB] += 1

				if winnerBA != -1:
					self.botScores[winnerBA] += 1
		'''

		trainGames = 10
		for a in range(NNPy.main.botCount):
			bot = tictactoe.botNN(self.botGame, NNPy.main.botList[a])

			for i in range(trainGames):
				randomWinnerFirst = self.botGame.playGameAgainstBots(bot, self.botGame.botRandom)
				if randomWinnerFirst == 1:
					self.botScores[a] += 3
				elif randomWinnerFirst == 0:
					self.botScores[a] += 1

				randomWinnerSecond = self.botGame.playGameAgainstBots(self.botGame.botRandom, bot)
				if randomWinnerSecond == -1:
					self.botScores[a] += 3
				elif randomWinnerSecond == 0:
					self.botScores[a] += 1


		print(self.botScores, round(sum(self.botScores)/(2*3*trainGames*NNPy.main.botCount), 2))
	

		random.seed(pygame.time.get_ticks())

		# Remove worst half of bots
		newBotList = []
		for i in range(int(NNPy.main.botCount/2)):
			bestBotIndex = self.botScores.argmax()	# Get best bots index
			self.botScores[bestBotIndex] = -1	# Don't make bot get picked again

			bestBot = NNPy.main.botList[bestBotIndex]
			bestBotChild = bestBot.getChild(0.05)

			# Add bot and its child to bot list
			newBotList.append(bestBot)
			newBotList.append(bestBotChild)		

		NNPy.main.botList = newBotList

		# Reset bot scores
		self.botScores.fill(0)




		# Play best bot against a random bot and store win rate in graph
		# Bot needs to play both first, and second
		testGames = 10	# Test games per type
		firstWins = 0
		secondWins = 0

		for i in range(testGames):
			bot = tictactoe.botNN(self.botGame, NNPy.main.botList[0])
			randomWinnerFirst = self.botGame.playGameAgainstBots(bot, self.botGame.botRandom)
			if randomWinnerFirst == 1:
				firstWins += 1

			randomWinnerSecond = self.botGame.playGameAgainstBots(self.botGame.botRandom, bot)
			if randomWinnerSecond == -1:
				secondWins += 1

		self.randomWinRate.append((firstWins + secondWins) / (2*testGames))
		self.randomWinRateFirst.append(firstWins/testGames)
		self.randomWinRateSecond.append(secondWins/testGames)


	def update(self, mouseState, dt):	

		if (self.autoTrain):
			for i in range(math.floor(self.autoTrainSpeed)):
				self.train()

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

		text = self.font.render(str(NNPy.main.botList[0].structure), True, (0,0,0))
		NNPy.main.screen.blit(text, [1050,580])

		text = self.font.render(str(NNPy.main.botList[0].LEARNING_RATE), True, (0,0,0))
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
		self.train()

	def train_10x(self):
		print("Training 10x")
		for i in range(10):
			self.train()

	def train_100x(self):
		print("Training 100x")
		for i in range(100):
			self.train()

	def toggleAutoTrain(self):
		self.autoTrain = not self.autoTrain
		return "Train: " + str(self.autoTrain)