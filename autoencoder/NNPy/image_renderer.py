

import math, pygame, numpy as np

class Image_Renderer():
	def __init__(self, screen, rect):

		self.screen = screen
		self.rect = rect
		self.colour = (230,230,230)		# Background colour

		self.size = (28,28)

		self.imageData = []


	def setImageData(self, imageData):
		self.imageData = imageData


	def render(self):

		pygame.draw.rect(self.screen, (230,230,230), self.rect)
		pygame.draw.rect(self.screen, (0,0,0), self.rect, 3)

		if len(self.imageData) != 0:
			pixelRect = pygame.Rect(0, 0, math.ceil(self.rect.w/28), math.ceil(self.rect.h/28))
			for x in range(self.size[0]):
				for y in range(self.size[1]):
					pixelRect.topleft = (
						self.rect.x + math.ceil(x*(self.rect.w/28)), 
						self.rect.y + math.ceil(y*(self.rect.h/28))
					)
					pixelColour = [int( (self.imageData[y*28+x]+1)/2 * 255 )] * 3

					pygame.draw.rect(self.screen, pixelColour, pixelRect)