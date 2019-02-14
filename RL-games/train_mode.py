
import sys, math, pygame, random, numpy as np
import NNPy, tictactoe, connect4


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



		self.botGame = connect4.Connect4Game()
		self.botScores = np.zeros(NNPy.main.botCount, dtype="int16")

		self.trainGames = 32	# Numbers of games each bot plays per generation
		self.trainBotNum = 0	# Current bot being trained for this particular train (bots are cycled through each train)



	def train(self):
		pass
		
		'''
		# === Score bots based on their ability to beat other bots in population
		# Play games between each possible pair of bots
		for a in range(NNPy.main.botCount):
			for b in range(a):
				# Play a whole game when a is player 1, and b is player -1, and then the other way around

				botA = connect4.botNN(self.botGame, NNPy.main.botList[a])
				botB = connect4.botNN(self.botGame, NNPy.main.botList[b])

				winnerAB = self.botGame.playGameAgainstBots(botA, botB)
				winnerBA = self.botGame.playGameAgainstBots(botB, botA)

				if winnerAB == 1:
					self.botScores[a] += 3
				elif winnerAB == -1:
					self.botScores[b] += 3
				else:
					self.botScores[a] += 1
					self.botScores[b] += 1

				if winnerBA == 1:
					self.botScores[b] += 3
				elif winnerBA == -1:
					self.botScores[a] += 3
				else:
					self.botScores[a] += 1
					self.botScores[b] += 1


		print(self.botScores)
		
		bestBotNN = NNPy.main.botList[self.botScores.argmax()]
		'''
		#bestBotNN = NNPy.main.botList[0]

		
		# === Score bots based on their ability to beat a random opponent
		if (self.trainBotNum < NNPy.main.botCount):		# Get the performance of a single bot

			bot = connect4.botNN(self.botGame, NNPy.main.botList[self.trainBotNum])		# Bot to train
			trainBot = self.botGame.botGood		# Bot to train against

			for i in range(self.trainGames):
				randomWinnerFirst = self.botGame.playGameAgainstBots(bot, trainBot)
				if randomWinnerFirst == 1:
					self.botScores[self.trainBotNum] += 3
				elif randomWinnerFirst == 0:
					self.botScores[self.trainBotNum] += 1

				randomWinnerSecond = self.botGame.playGameAgainstBots(trainBot, bot)
				if randomWinnerSecond == -1:
					self.botScores[self.trainBotNum] += 3
				elif randomWinnerSecond == 0:
					self.botScores[self.trainBotNum] += 1

			self.trainBotNum += 1


		else:	# Create next generation of bots
			self.trainBotNum = 0

			print(
				self.botScores, 
				round(sum(self.botScores)/(2*3*self.trainGames*NNPy.main.botCount), 2), 
				round(self.botScores.max()/(2*3*self.trainGames),2)
			)

			bestBotNN = NNPy.main.botList[self.botScores.argmax()]
			

			
			# === Remove worst half of bots and replace with children of best half
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
			

			'''
			# === Apply mutations to worst half of bots ===
			for i in range(int(NNPy.main.botCount/2)):
				worstBotIndex = self.botScores.argmin()	# Get worst bots index

				# Don't make bot get picked again by giving it a max score
				self.botScores[worstBotIndex] = NNPy.main.botCount * 2

				# Replace bad bot with a child of itself
				NNPy.main.botList[worstBotIndex] = NNPy.main.botList[worstBotIndex].getChild(0.05)

			# Reset bot scores
			self.botScores.fill(0)
			'''

			
			# Play best bot against a random bot and store win rate in graph
			# Bot needs to play both first, and second
			testGames = 16	# Test games per type
			firstWins = 0
			secondWins = 0
			testBot = self.botGame.botGood
			bestBot = connect4.botNN(self.botGame, bestBotNN)
			for i in range(testGames):
				randomWinnerFirst = self.botGame.playGameAgainstBots(bestBot, testBot)
				if randomWinnerFirst == 1:
					firstWins += 1

				randomWinnerSecond = self.botGame.playGameAgainstBots(testBot, bestBot)
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


		# Draw training bot number
		text = self.font.render("Bot Num: " + str(self.trainBotNum), True, (0,0,0))
		NNPy.main.screen.blit(text, [1050,400])
		



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