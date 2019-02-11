
import sys, math, pygame, random, numpy as np
import NNPy, tictactoe


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

		]


		self.game = tictactoe.TicTacToeGame()




	def update(self, mouseState, dt):	

		self.game.update(mouseState)


	def render(self, mouseState):

		self.game.render(mouseState)

		#text = self.font.render("TESTING MODE :)", True, (0,0,0))
		#self.screen.blit(text, [760,105])

		#text = self.font.render("Results: " + self.testOutput, True, (0,0,0))
		#self.screen.blit(text, [760,155])






