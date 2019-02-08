

import math, pygame

class Slider():
	def __init__(self, screen, rect, action, orientation="vertical"):

		self.screen = screen
		self.rect = rect
		self.colour_background = (150,150,100)		# Background colour
		self.colour_slider = (100,100,100)			# Colour of sliding part

		self.action = action	# Function that runs 

		# Stores which direction slider slides (left to right or top to bottom)
		self.orientation = orientation

		self.slideValue = 0.5

	def update(self, mouseState):	# Assumes button has been pressed and updates accordingly

		if mouseState[2]:
			if self.rect.collidepoint(mouseState[3]):

				if (self.orientation == "horizontal"):
					self.slideValue = (mouseState[0][0] - self.rect.x) / self.rect.w
				elif (self.orientation == "vertical"):
					self.slideValue = (self.rect.y + self.rect.h - mouseState[0][1]) / self.rect.h

				self.slideValue = max(self.slideValue, 0)
				self.slideValue = min(self.slideValue, 1)

				self.action(self.slideValue)


	def render(self):

		# Draw background
		pygame.draw.rect(self.screen, self.colour_background, self.rect)
		pygame.draw.rect(self.screen, (0,0,0), self.rect, 2)


		# Draw slider part
		sliderRect = self.rect.copy()

		if (self.orientation == "horizontal"):
			sliderRect.w = round(sliderRect.w * self.slideValue)
		elif (self.orientation == "vertical"):
			sliderRect.h = round(sliderRect.h * self.slideValue)
			sliderRect.bottom = self.rect.bottom

		pygame.draw.rect(self.screen, self.colour_slider, sliderRect)
		pygame.draw.rect(self.screen, (0,0,0), sliderRect, 2)

