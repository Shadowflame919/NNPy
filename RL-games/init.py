
'''

	Compresses and uncompresses an image :)
	

'''

import sys, math, random, pygame, json, numpy as np
import NNPy
import train_mode, test_mode, tictactoe, connect4
pygame.init()


print("Beginning Program")


def init(self):
	print("Running Setup")


	## Create bots ##
	'''
		VS Random
			Method 1:
				Create list of bots
				Play n games against random and rank them
				Remove worst half
				Each bot has a child
				Repeat
			This is basic artifical selection

			Method 2:
				Have a single bot
				Make change, check if better than parent
				If not sucessful, revert change and try again
				If sucessful, replace parent and repeat
			This is hillclimbing I believe



		Bots that pick their most favoured room every time would be more stable, yet be less likely to improve over time

		This is because a bot that only needs to rank their best move would not care for how it ranks all other moves
		Therefore the bot would not care about ranking its 2nd best move a meaningful value
		If the bot was picking move A, but should be picking move B, it would perform worse if rank B was rated last, 
		than if it was rated 2nd and had a small chance of being picked. Therefore bots that rank move B 2nd would be selected
		over bots that rank move B worse than second. Bots who rank move B 2nd are also more likely to have a mutation that causes them
		to rank move B 1st. This provides a less steep evolutionary landscape for the bot, as it can mutate from rank B = last, 
		to rank B = 2nd, to rank B = 1st in an artificially selected manner. 

	'''

	self.botCount = 16
	self.botList = []
	for i in range(self.botCount):
		newBot = NNPy.NN([42,32,1], 0.001)
		self.botList.append(newBot)



	#print(self.nn)



	# Create and initialise the modes
	self.modeList = [mode(self) for mode in mainParams["modeList"]]
	self.modeNum = 0
	self.mode = self.modeList[self.modeNum]






NNPy.Main.init = init

mainParams = {
	"modeList": [
		train_mode.Train_Mode,
		test_mode.Test_Mode
	]
}

NNPy.main = NNPy.Main(mainParams)
NNPy.main.init()
while True:
	NNPy.main.update()
	NNPy.main.render()


