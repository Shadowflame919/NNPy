
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
					return self.move(col)
					

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
		
		for row in range(6):
			if self.board[col][row] == 0:

				self.board[col][row] = self.turn
				self.turnNum += 1			

				winner = self.doesMoveEndGame(col, row, self.turn)

				if winner == None:
					self.turn *= -1
				
				return winner


	def doesMoveEndGame(self, col, row, turn):	# Returns whether moving in a particular position ends the game
		# Returns 1, 0, or -1 if this particular move ends the game, otherwise return None

		# Check whether piece creates vertical 4 in a row
		count = 1
		for rowBelow in range(row-1, max(row-4,-1), -1):
			if self.board[col][rowBelow] == turn:
				count += 1
				if count == 4:
					return turn
		
		
		# Check if piece creates horizontal 4 in a row
		count = 1

		for colRight in range(col+1, min(col+4,7)):
			if self.board[colRight][row] == turn:
				count += 1
			else:
				break	

		for colLeft in range(col-1, max(col-4,-1), -1):
			if self.board[colLeft][row] == turn:
				count += 1
			else:
				break

		if count >= 4:
			return turn


		# Check whether piece create diagonal TL to BR
		count = 1

		for offset in range(1, 4):		# Bottom right of piece
			if col+offset==7 or row-offset==-1 or self.board[col+offset][row-offset] != turn:
				break
			count += 1

		for offset in range(1, 4):	# Top left of piece
			if col-offset==-1 or row+offset==6 or self.board[col-offset][row+offset] != turn:
				break
			count += 1

		if count >= 4:
			return turn


		# Check whether piece create diagonal TR to BL
		count = 1

		for offset in range(1, 4):		# Bottom left of piece
			if col-offset==-1 or row-offset==-1 or self.board[col-offset][row-offset] != turn:
				break
			count += 1

		for offset in range(1, 4):	# Top right of piece
			if col+offset==7 or row+offset==6 or self.board[col+offset][row+offset] != turn:
				break
			count += 1

		if count >= 4:
			return turn

		# Game is tie since all squares are filled
		if self.turnNum == 42:
			return 0

		# Move does NOT end game
		return None



	def isValidMove(self, col):		# Returns whether a move is valid
		return self.board[col][5] == 0

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
			selectedMove = -1
			if self.turn == 1:
				selectedMove = botA.getMove()
			else:
				selectedMove = botB.getMove()

			winner = self.move(selectedMove)

			if winner != None:
				self.resetGame()
				return winner


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

		# Check for winning placements
		for col in range(7):
			# If move has no free slots, skip this column
			if self.game.board[col][5] != 0:
				continue

			# Find row which piece would be placed in this column
			row = 0
			for row in range(6):
				if self.game.board[col][row] == 0:
					break

			winner = self.game.doesMoveEndGame(col, row, self.game.turn)

			if winner == None:
				continue
			return col


		# Return a random 'safe' move
		self.validMoves.fill(0)
		self.validMoveCount = 0

		
		# Check for blocking placements
		for col in range(7):
			# If move has no free slots, skip this column
			if self.game.board[col][5] != 0:
				continue

			# Find row which piece would be placed in this column
			row = 0
			for row in range(6):
				if self.game.board[col][row] == 0:
					break


			# If piece blocks opponent, always play here
			winner = self.game.doesMoveEndGame(col, row, -self.game.turn)
			if winner == -self.game.turn:
				return col


			# Add moves that don't cause opponent to win next turn (safe moves)
			if row == 5 or self.game.doesMoveEndGame(col, row+1, -self.game.turn) == None:
				self.validMoves[self.validMoveCount] = col
				self.validMoveCount += 1


		# Pick a random move from this list of 'safe' moves
		if self.validMoveCount > 0:
			return self.validMoves[ random.randint(0,self.validMoveCount-1) ]


		# Return a random move
		self.validMoves.fill(0)
		self.validMoveCount = 0
		for i in range(7):		# Add all possible moves to valid moves list
			if self.game.isValidMove(i):
				self.validMoves[self.validMoveCount] = i
				self.validMoveCount += 1
		return self.validMoves[ random.randint(0,self.validMoveCount-1) ]
