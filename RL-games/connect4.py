
import sys, math, pygame, random, numpy as np
import NNPy


class Connect4Game():
	def __init__(self):
		
		self.renderRect = pygame.Rect(100, 100, 630, 540)

		self.turn = 1
		self.turnNum = 0	# Keeps track of number of turns played (useful for detecting game over)

		self.board = np.array([np.zeros(6, dtype="int8") for col in range(7)])

		self.botRandom = botRandom(self)
		self.botGood = botGood(self)

	def update(self, mouseState):	

		
		if mouseState[1]:
			rect = self.renderRect
			for col in range(7):
				colRect = pygame.Rect(
					rect.x + (col/7)*rect.w,
					rect.y,
					rect.w/7,
					rect.h
				)
			
				if colRect.collidepoint(mouseState[0]) and self.board[col][5] == 0:
					self.move(col)

					if self.isGameOver():
						print("Winner: ", -self.turn)
						self.resetGame()
					elif self.isGameTie():
						print("Tie")
						self.resetGame()

			'''
			test = pygame.Rect(0,0,50,50)
			if test.collidepoint(mouseState[0]):
				time = pygame.time.get_ticks()
				games = 0
				tied = False
				while not tied:
					games += 1
					while True:
						x = random.randint(0,6)
						if self.board[x][5] == 0:
							self.move(x)

							if self.isGameOver():
								print("Winner: ", -self.turn)
								self.resetGame()
								break
							elif self.isGameTie():
								print("Tie")
								tied = True
								break
				print("Games", games, pygame.time.get_ticks()-time)'''



	def render(self, mouseState):

		rect = self.renderRect

		pygame.draw.rect(NNPy.main.screen, (255,255,255), rect)
		pygame.draw.rect(NNPy.main.screen, (0,0,0), rect, 2)

		for row in range(1,6):
			pygame.draw.line(NNPy.main.screen, (0,0,0), (rect.left, rect.y + rect.h*(row/6)), (rect.right, rect.y + rect.h*(row/6)), 2)

		for col in range(1,7):
			pygame.draw.line(NNPy.main.screen, (0,0,0), (rect.x + rect.w*(col/7), rect.top), (rect.x + rect.w*(col/7), rect.bottom), 2)
		
		for col in range(7):
			for row in range(6):
				pieceRect = pygame.Rect(
					rect.x + (col/7)*rect.w,
					rect.y + ((5-row)/6)*rect.h,
					rect.w/7,
					rect.h/6
				)

				if self.board[col][row] != 0:
					pieceColour = (255,0,0)
					if self.board[col][row] == -1:
						pieceColour = (0,0,0)
					
					pygame.draw.circle(NNPy.main.screen, pieceColour, pieceRect.center, int(pieceRect.w/2.5), 0)


	def move(self, col):	# Makes move in a particular column and switches turn
		
		for row in range(7):
			if self.board[col][row] == 0:
				self.board[col][row] = self.turn
				self.turn *= -1
				self.turnNum += 1
				break

	def isValidMove(self, col):		# Returns whether a move is valid
		return self.board[col][5] == 0

	def isGameOver(self):	# Returns whether game is over based on current boardstate
		# test all win states somehow
		
		# First check for vertical wins
		# This requires first a test in the middle piece, as this determines whether a win is even possible
		# Test for a piece in the higher positions first, as these are less likely to have pieces
		for col in range(7):
			comparePiece = self.board[col][3]	# Player (1 or -1) who may potentially have 4 in a row in this column
			if comparePiece==0:		# If this square is zero, player cannot have a 4 in a row here
				continue

			# loop through col, counting for sequence of 4
			count = 0
			for row in range(6):
				if self.board[col][row] == comparePiece:
					count += 1
					if count == 4:
						return True
				else:
					count = 0


		# Test for 4 in a row in each row
		for row in range(6):
			comparePiece = self.board[3][row]	# Player (1 or -1) who may potentially have 4 in a row in this row
			if comparePiece==0:		# If this square is zero, player cannot have a 4 in a row here
				continue

			# loop through row, counting for sequence of 4
			count = 0
			for col in range(7):
				if self.board[col][row] == comparePiece:
					count += 1
					if count == 4:
						return True
				else:
					count = 0

		
		# Test for 4 in a row in each diagonal
		# Testing along 4th row up should test for each diagonal
		for col in range(7):
			comparePiece = self.board[col][3]
			if comparePiece==0:		# If this square is zero, player cannot have a 4 through any diagonal here
				continue

			# test for TL->BR diag
			if col <= 5:
				count = 0

				# start diagonal at top/left of board
				startCol = 0
				startRow = 5
				if col <= 2: 
					startRow = 3 + col
				else:
					startCol = col - 2
				
				# count for pieces down and right
				offset=0
				while True:
					if self.board[startCol+offset][startRow-offset] == comparePiece:
						count += 1
						if count == 4:
							return True
					else:
						count = 0
					offset += 1
					if startCol+offset==7 or startRow-offset==-1:	# If the next piece is off the board, break loop
						break

			# test for TR->BL diag
			if col >= 1:
				count = 0

				# start diagonal at top/right of board
				startCol = 6
				startRow = 5
				if col >= 4: 
					startRow = 9 - col
				else:
					startCol = col + 2
				
				# count for pieces down and left
				offset=0
				while True:
					if self.board[startCol-offset][startRow-offset] == comparePiece:
						count += 1
						if count == 4:
							return True
					else:
						count = 0
					offset += 1
					if startCol+offset==-1 or startRow-offset==-1:	# If the next piece is off the board, break loop
						break
		

	def isGameTie(self):
		return self.turnNum == 42

	def resetGame(self):
		self.board.fill(0)
		self.turn = 1
		self.turnNum = 0

	def getBotInput(self):		# Gets the bot input for a particular board state (bot is always player 1)
		board = self.board.flatten()
		if self.turn == -1:
			board *= -1
		return board


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
		self.moveRanks = np.array([-1.]*7)

	def getMove(self):

		for i in range(7):	# for all possible moves
			if self.game.isValidMove(i):		# if move is valid
				# gets bots rating of this move

				# get input for NN
				boardInput = self.game.getBotInput()

				# make future move in this board input
				replacementIndex = i*6
				while boardInput[replacementIndex] != 0:
					replacementIndex += 1
				boardInput[replacementIndex] = 1

				self.moveRanks[i] = self.NN.getOutput(boardInput)[0]

		selectedMove = self.moveRanks.argmax()	# get highest rated move

		# Select highest rated move with certain probability
		#while True:
		#	selectedMove = self.moveRanks.argmax()	# get highest rated move
		#	if random.random() < 0.95 or self.moveRanks.max() == -2:	# pick move with 90% probability (or if all moves have been selected, select this move)
		#		break
		#	self.moveRanks[selectedMove] = -2
		
		self.moveRanks.fill(-1)		# reset move ranks

		return selectedMove



class botRandom():
	def __init__(self, game):
		self.game = game
		self.validMoves = np.zeros(7, dtype="int8")
		self.validMoveCount = 0

	def getMove(self):
		self.validMoves.fill(0)
		self.validMoveCount = 0

		for i in range(7):		# for all possible moves
			if self.game.isValidMove(i):
				self.validMoves[self.validMoveCount] = i
				self.validMoveCount += 1

		return self.validMoves[ random.randint(0,self.validMoveCount-1) ]



class botGood():
	def __init__(self, game):
		self.game = game
		self.validMoves = np.zeros(7, dtype="int8")
		self.validMoveCount = 0

	def getMove(self):
		self.validMoves.fill(0)
		self.validMoveCount = 0

		for i in range(7):		# for all possible moves
			if self.game.isValidMove(i):
				self.validMoves[self.validMoveCount] = i
				self.validMoveCount += 1

				# If row has 3 in a row, block column instead of a random move
				count = 0
				for row in range(5,-1,-1):
					if self.game.board[i][row] == -self.game.turn:
						count += 1
						if count == 3:
							return i
					elif self.game.board[i][row] == self.game.turn:
						break

		return self.validMoves[ random.randint(0,self.validMoveCount-1) ]


