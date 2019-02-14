
import sys, math, pygame, random, numpy as np
import NNPy, tictactoe, connect4


class Test_Mode():
	def __init__(self, main):
		self.mode = "test"

		self.font = pygame.font.SysFont(None, 30)
		#textrect = text.get_rect()
		#textrect.centerx = screen.get_rect().centerx
		#textrect.centery = screen.get_rect().centery

		self.testImage = 0
		self.imageRendererInput = NNPy.Image_Renderer(NNPy.main.screen, pygame.Rect(50,100,280,280))
		self.imageRendererOutput = NNPy.Image_Renderer(NNPy.main.screen, pygame.Rect(50,400,280,280))

		# The image renderer used for tweaking the sliders manually
		self.imageRendererCustom = NNPy.Image_Renderer(NNPy.main.screen, pygame.Rect(900,100,280,280))

		self.buttonList = [
			NNPy.Button(NNPy.main.screen, pygame.Rect(780, 20, 150, 30), "Download", 30, NNPy.main.downloadNetwork),
			NNPy.Button(NNPy.main.screen, pygame.Rect(950, 20, 150, 30), "Upload", 30, NNPy.main.uploadNetwork),

			NNPy.Button(NNPy.main.screen, pygame.Rect(800, 400, 150, 30), "Random Move", 30, self.randomMove),
			NNPy.Button(NNPy.main.screen, pygame.Rect(800, 450, 150, 30), "Good Move", 30, self.goodMove),
			NNPy.Button(NNPy.main.screen, pygame.Rect(800, 500, 150, 30), "Bot Move", 30, self.botMove),
			NNPy.Button(NNPy.main.screen, pygame.Rect(800, 550, 150, 30), "Reset Game", 30, self.resetGame),

			NNPy.Button(NNPy.main.screen, pygame.Rect(1000, 400, 150, 30), "Test Random", 30, self.testBotAgainstRandom),
			NNPy.Button(NNPy.main.screen, pygame.Rect(1000, 450, 150, 30), "Test Good", 30, self.testBotAgainstGood),
			NNPy.Button(NNPy.main.screen, pygame.Rect(1000, 500, 150, 30), "Good VS Rand", 30, self.testGoodVSRandom),

		]


		self.game = connect4.Connect4Game()

		self.gameOver = False


		self.testResults = ""


	def update(self, mouseState, dt):	

		self.gameOver = self.game.isGameOver() or self.game.isGameTie()

		if not self.gameOver:
			self.game.update(mouseState)


	def render(self, mouseState):

		self.game.render(mouseState)

		if self.gameOver:
			winner = "Red (1)" if (self.game.turn==-1) else "Black (-1)"
			text = self.font.render("Winner: " + winner, True, (0,0,0))
			NNPy.main.screen.blit(text, [760,105])

		text = self.font.render(self.testResults, True, (0,0,0))
		NNPy.main.screen.blit(text, [760,155])


	def botMove(self):

		if not self.gameOver:

			bot = connect4.botNN(self.game, NNPy.main.botList[0])
			move = bot.getMove()
			self.game.move(move)

			if self.game.isGameOver():
				print("Winner: ", -self.game.turn)
			elif self.game.isGameTie():
				print("Tie")


	def randomMove(self):

		if not self.gameOver:

			move = self.game.botRandom.getMove()
			self.game.move(move)

			if self.game.isGameOver():
				print("Winner: ", -self.game.turn)
			elif self.game.isGameTie():
				print("Tie")


	def goodMove(self):

		if not self.gameOver:

			move = self.game.botGood.getMove()
			self.game.move(move)

			if self.game.isGameOver():
				print("Winner: ", -self.game.turn)
			elif self.game.isGameTie():
				print("Tie")


	def resetGame(self):
		self.game.resetGame()



	def testBotAgainstGood(self):	# Tests the best bot against a good bot and displays result on screen
		testGames = 100
		firstWins = 0
		secondWins = 0
		ties = 0

		botA = connect4.botNN(self.game, NNPy.main.botList[0])
		botB = self.game.botGood
		for i in range(testGames):
			randomWinnerFirst = self.game.playGameAgainstBots(botA, botB)
			if randomWinnerFirst == 1:
				firstWins += 1
			elif randomWinnerFirst == 0:
				ties += 1

			randomWinnerSecond =  self.game.playGameAgainstBots(botB, botA)
			if randomWinnerSecond == -1:
				secondWins += 1
			elif randomWinnerSecond == 0:
				ties += 1

		self.testResults = str(firstWins) + ", " + str(secondWins) + ", " + str(ties) + " (" + str(testGames) + ") - VS Good"
		#print(firstWins, secondWins, ties)


	def testBotAgainstRandom(self):	# Tests the best bot against a good bot and displays result on screen
		testGames = 100
		firstWins = 0
		secondWins = 0
		ties = 0

		botA = connect4.botNN(self.game, NNPy.main.botList[0])
		botB = self.game.botRandom
		for i in range(testGames):
			randomWinnerFirst = self.game.playGameAgainstBots(botA, botB)
			if randomWinnerFirst == 1:
				firstWins += 1
			elif randomWinnerFirst == 0:
				ties += 1

			randomWinnerSecond =  self.game.playGameAgainstBots(botB, botA)
			if randomWinnerSecond == -1:
				secondWins += 1
			elif randomWinnerSecond == 0:
				ties += 1

		self.testResults = str(firstWins) + ", " + str(secondWins) + ", " + str(ties) + " (" + str(testGames) + ") - VS Random"
		#print(firstWins, secondWins, ties)

	
	def testGoodVSRandom(self):
		testGames = 100
		firstWins = 0
		secondWins = 0
		ties = 0

		botA = self.game.botGood
		botB = self.game.botRandom

		for i in range(testGames):
			randomWinnerFirst = self.game.playGameAgainstBots(botA, botB)
			if randomWinnerFirst == 1:
				firstWins += 1
			elif randomWinnerFirst == 0:
				ties += 1

			randomWinnerSecond =  self.game.playGameAgainstBots(botB, botA)
			if randomWinnerSecond == -1:
				secondWins += 1
			elif randomWinnerSecond == 0:
				ties += 1

		self.testResults = str(firstWins) + ", " + str(secondWins) + ", " + str(ties) + " (" + str(testGames) + ") - Good VS Random"
