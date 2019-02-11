
import sys, math, pygame, random, numpy as np
import NNPy


class TicTacToeGame():
	def __init__(self):
		
		self.renderRect = pygame.Rect(100, 100, 500, 500)

		self.turn = 1
		self.turnNum = 0	# Keeps track of number of turns played (useful for detecting game over)
		self.board = np.zeros(9, dtype="int8")	# Top left down to bottom right

		self.botRandom = botRandom(self)
		self.botGood = botGood(self)

	def update(self, mouseState):	

		if mouseState[1]:
			for i,k in enumerate(self.board):
				squareRect = pygame.Rect(
					self.renderRect.x + (i//3)*(self.renderRect.w/3),
					self.renderRect.y + (i%3)*(self.renderRect.h/3),
					self.renderRect.w/3,
					self.renderRect.h/3
				)
				
				if squareRect.collidepoint(mouseState[0]) and k == 0:
					self.move(i)



	def render(self, mouseState):

		pygame.draw.rect(NNPy.main.screen, (255,255,255), self.renderRect)
		pygame.draw.rect(NNPy.main.screen, (0,0,0), self.renderRect, 2)

		rect = self.renderRect
		pygame.draw.line(NNPy.main.screen, (0,0,0), (rect.x + rect.w/3, rect.top), (rect.x + rect.w/3, rect.bottom), 5)
		pygame.draw.line(NNPy.main.screen, (0,0,0), (rect.x + 2*rect.w/3, rect.top), (rect.x + 2*rect.w/3, rect.bottom), 5)
		pygame.draw.line(NNPy.main.screen, (0,0,0), (rect.left, rect.y + rect.w/3), (rect.right, rect.y + rect.w/3), 5)
		pygame.draw.line(NNPy.main.screen, (0,0,0), (rect.left, rect.y + 2*rect.w/3), (rect.right, rect.y + 2*rect.w/3), 5)

		for i,k in enumerate(self.board):
			squareRect = pygame.Rect(
				rect.x + (i//3)*(rect.w/3) + rect.w/20,
				rect.y + (i%3)*(rect.h/3) + rect.h/20,
				rect.w/3 - 2*rect.w/20,
				rect.h/3 - 2*rect.h/20
			)

			if k == 1:
				pygame.draw.line(NNPy.main.screen, (0,0,0), squareRect.topleft, squareRect.bottomright, 10)
				pygame.draw.line(NNPy.main.screen, (0,0,0), squareRect.topright, squareRect.bottomleft, 10)
			elif k == -1:
				pygame.draw.circle(NNPy.main.screen, (0,0,0), squareRect.center, int(squareRect.w/2), 0)
				pygame.draw.circle(NNPy.main.screen, (255,255,255), squareRect.center, int(squareRect.w/2.2), 0)


		#text = self.font.render("TESTING MODE :)", True, (0,0,0))
		#self.screen.blit(text, [760,105])

		#text = self.font.render("Results: " + self.testOutput, True, (0,0,0))
		#self.screen.blit(text, [760,155])



	def move(self, squareNum):	# Makes move in a particular square and switches turn
		self.board[squareNum] = self.turn
		self.turn *= -1
		self.turnNum += 1

	def isValidMove(self, squareNum):	# Returns whether a move is valid
		return (self.board[squareNum] == 0)

	def isGameOver(self):	# Returns whether game is over based on current boardstate
		# test all 3+3+2=8 win states

		if (0 != self.board[0] == self.board[1] == self.board[2] 		# row 1
		or 0 != self.board[3] == self.board[4] == self.board[5] 		# row 2
		or 0 != self.board[6] == self.board[7] == self.board[8] 		# row 3
		or 0 != self.board[0] == self.board[3] == self.board[6] 		# col 1
		or 0 != self.board[1] == self.board[4] == self.board[7] 		# col 2
		or 0 != self.board[2] == self.board[5] == self.board[8] 		# col 3
		or 0 != self.board[0] == self.board[4] == self.board[8] 		# diag TL to BR
		or 0 != self.board[2] == self.board[4] == self.board[6]):	# diag TR to BL
			return True

	def isGameTie(self):
		return self.turnNum == 9
			
	def resetGame(self):
		self.board.fill(0)
		self.turn = 1
		self.turnNum = 0

	def getBotInput(self):		# Gets the bot input for a particular board state (bot is always player 1)
		board = self.board.copy()
		if self.turn == -1:
			board *= -1
		return board


	def playGame(self, a, b):		# Plays a whole game between two bots (a and b are indexes) and returns the winners index
		botA = botNN(NNPy.main.botList[a])
		botB = botNN(NNPy.main.botList[b])

		# Used for bots to loop through each future move and give it a value
		moveRanks = np.array([-1.]*9)

		while True:		# Loop each move

			# Play move
			if self.turn == 1:
				for i in range(9):	# for all possible moves
					if self.isValidMove(i):

						boardInput = self.getBotInput()
						boardInput[i] = 1

						moveRanks[i] = botA.getOutput(boardInput)[0]

				#print("A moveranks", moveRanks)
				selectedMove = moveRanks.argmax()
				self.move(selectedMove)
				#print("A moved in ", selectedMove)

				moveRanks.fill(-1)

			else:
				for i in range(9):	# for all possible moves
					if self.isValidMove(i):

						boardInput = self.getBotInput()
						boardInput[i] = 1

						moveRanks[i] = botB.getOutput(boardInput)[0]

				#print("B moveranks", moveRanks)
				selectedMove = moveRanks.argmax()
				self.move(selectedMove)
				#print("B moved in ", selectedMove)

				moveRanks.fill(-1)

			#print(self.board)

			#input()


			# First test if game is over
			if self.isGameOver():
				if self.turn == 1:
					self.resetGame()
					return b
				else:
					self.resetGame()
					return a

			# Then test if game is tie
			if self.isGameTie():
				self.resetGame()
				return -1



	def playGameAgainstBots(self, botA, botB):		# Plays a whole game between bot A and bot B, with bot A going first
		# Returns 1 if botA wins, -1 if botB wins, and 0 if tie

		

		while True:		# Loop each move

			# Play move
			if self.turn == 1:
				selectedMove = botA.getMove()
				self.move(selectedMove)
			else:
				selectedMove = botB.getMove()
				self.move(selectedMove)


			# First test if game is over
			if self.isGameOver():
				# Return 1 if botA wins, -1 if botB wins
				winner = -self.turn;
				self.resetGame()
				return winner

			# Then test if game is tie
			if self.isGameTie():
				self.resetGame()
				return 0	# Return 0 if game is tie


class botNN():
	def __init__(self, game, NN):
		self.game = game
		self.NN = NN

		# Used for bot to loop through each future move and give it a value
		self.moveRanks = np.array([-1.]*9)

	def getMove(self):

		for i in range(9):	# for all possible moves
			if self.game.isValidMove(i):		# if move is valid
				# gets bots rating of this move

				# get input for NN
				boardInput = self.game.getBotInput()
				boardInput[i] = 1

				self.moveRanks[i] = self.NN.getOutput(boardInput)[0]

		selectedMove = self.moveRanks.argmax()	# get highest rated move
		self.moveRanks.fill(-1)		# reset move ranks

		return selectedMove



class botRandom():
	def __init__(self, game):
		self.game = game
		self.validMoves = np.zeros(9, dtype="int8")
		self.validMoveCount = 0

	def getMove(self):
		self.validMoves.fill(0)
		self.validMoveCount = 0

		for i in range(9):		# for all possible moves
			if self.game.isValidMove(i):
				self.validMoves[self.validMoveCount] = i
				self.validMoveCount += 1

		return self.validMoves[ random.randint(0,self.validMoveCount-1) ]


class botGood():
	def __init__(self, game):
		self.game = game

	def getMove(self):

		board = self.game.board

		# Assume we are first
		if self.game.turnNum == 0:	# Play center first
			return 4

		if self.game.turnNum == 1:
			return 0

		if self.game.turnNum == 2:
			if board[1] == -1 or board[3] == -1 or board[5] == -1 or board[7] == -1:
				return 0

			if board[0] == -1:
				return 2
			if board[2] == -1:
				return 0
			if board[6] == -1:
				return 0
			if board[8] == -1:
				return 2

		if self.game.turnNum == 3:
			if board[1] == 1 or board[2] == 1 or board[5] == 1:
				return 6
			else:
				return 2

		if self.game.turnNum == 4:
			if board[0] == 1 and board[8] == 0:
				return 8
			if board[2] == 1 and board[6] == 0:
				return 6

		if self.game.turnNum == 5:
			if board[2] == -1 and board[1] == 0:
				return 1
			if board[6] == -1 and board[3] == 0:
				return 3

		return self.game.botRandom.getMove()

