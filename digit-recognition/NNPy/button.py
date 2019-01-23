

import math, pygame

class Button():
	def __init__(self, screen, rect, text, font, action):

		self.screen = screen
		self.rect = rect
		self.colour = (150,150,100)		# Background colour
		self.colour_hover = (100,100,100)

		self.action = action

		self.font = pygame.font.SysFont(None, font)
		self.setButtonText(text)


	def setButtonText(self, text):
		self.text = text
		self.textObject = self.font.render(str(self.text), True, (0,0,0))
		self.textRect = self.textObject.get_rect()
		self.textRect.center = self.rect.center

	def update(self, mouseState):	# Assumes button has been pressed and updates accordingly

		if mouseState[1]:
			if self.rect.collidepoint(mouseState[0]):
				newText = self.action()
				if newText != None:		# Change button text if function returns a value
					self.setButtonText(newText)
				mouseState[1] = False


	def render(self, mouseState):

		colour = self.colour
		if self.rect.collidepoint(mouseState[0]): 
			colour = self.colour_hover

		pygame.draw.rect(self.screen, colour, self.rect)
		pygame.draw.rect(self.screen, (0,0,0), self.rect, 3)

		# Render text inside button
		self.screen.blit(self.textObject, self.textRect)

