

import math, pygame
import button

class Graph():
	def __init__(self, screen, rect, title, data, minValue=0, maxValue=0):

		self.screen = screen
		self.rect = rect
		self.colour = (200,200,200)		# Background colour

		self.title = title

		self.font = pygame.font.SysFont(None, 30)


		self.data = data

		self.oldLength = len(self.data)

		# Initialise min/max values for the graph to render with
		self.minValue = minValue
		self.maxValue = maxValue
		for i in self.data:
			if i>self.maxValue:
				self.maxValue = i
			elif i<self.minValue:
				self.minValue = i

		self.scale = 0

		self.buttonList = [
			button.Button(self.screen, pygame.Rect(self.rect.right-40, self.rect.top+10, 30, 30), ">", 40, self.scale_up),
			button.Button(self.screen, pygame.Rect(self.rect.right-75, self.rect.top+10, 30, 30), "<", 40, self.scale_down)
		]


	def update(self, mouseState):	# Starts the main update loop

		# Buttons for changing scale
		for button in self.buttonList:
			button.update(mouseState)

		# Test if items have been added to data and update min/max accordingly
		if self.oldLength < len(self.data):
			for i in self.data[self.oldLength:]:
				if i>self.maxValue:
					self.maxValue = i
				elif i<self.minValue:
					self.minValue = i




	def render(self, mouseState):
		pygame.draw.rect(self.screen, self.colour, self.rect)
		pygame.draw.rect(self.screen, (0,0,0), self.rect, 3)

		if len(self.data) >= 2 and self.maxValue != self.minValue:
			pygame.draw.lines(self.screen, (255,0,0), False, [
				[
					self.rect.left + self.rect.w*(k/(len(self.data)-1)), 
					self.rect.bottom - self.rect.h*((i-self.minValue)/(self.maxValue-self.minValue))
				] for k,i in enumerate(self.data)
			], 2)

		# Render min/max value
		text = self.font.render(str(self.maxValue), True, (0,0,0))
		self.screen.blit(text, [self.rect.left+5, self.rect.top+5])
		text = self.font.render(str(self.minValue), True, (0,0,0))
		self.screen.blit(text, [self.rect.left+5, self.rect.bottom-text.get_rect().h-5])

		# Render No. of items
		text = self.font.render(str(len(self.data)), True, (0,0,0))
		self.screen.blit(text, [self.rect.right-text.get_rect().w-5, self.rect.bottom-text.get_rect().h-5])

		# Render title
		text = self.font.render(str(self.title), True, (0,0,0))
		self.screen.blit(text, [self.rect.centerx-text.get_rect().w/2, self.rect.top+5])

		# Render scale
		text = self.font.render(str(10**self.scale)+"x", True, (0,0,0))
		self.screen.blit(text, [self.rect.right-text.get_rect().w-85, self.rect.top+15])


		for button in self.buttonList:
			button.render(mouseState)


	def scale_up(self):
		self.scale += 1

	def scale_down(self):
		if self.scale > 0:
			self.scale -= 1
