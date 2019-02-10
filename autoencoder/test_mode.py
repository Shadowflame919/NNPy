
import sys, math, pygame, random, numpy as np
import NNPy


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

			NNPy.Button(NNPy.main.screen, pygame.Rect(350, 110, 30, 30), "<", 40, self.prevTestImage),
			NNPy.Button(NNPy.main.screen, pygame.Rect(385, 110, 30, 30), ">", 40, self.nextTestImage),

			NNPy.Button(NNPy.main.screen, pygame.Rect(730, 100, 150, 30), "Use Image", 30, self.useImage),

			#NNPy.Button(self.screen, pygame.Rect(685, 250, 100, 30), "Output", 30, self.getDigitOutput)
		]


		

		# Add a bunch of sliders that alter the compressed tensor
		self.sliderCount = min(NNPy.main.nn.structure)
		self.sliderList = []
		self.compressedInput = np.array([0.0]*self.sliderCount)
		for i in range(self.sliderCount):
			#self.compressedInput.append(0)

			sliderRows = 3
			sliderRect = pygame.Rect(
				800 + 20 * (i%round(self.sliderCount/sliderRows)), 
				420 + 95 * math.floor(sliderRows*i/self.sliderCount), 
			15, 80)

			def sliderFunction(x,index=i):
				self.compressedInput[index] = 2*x-1


			newSlider = NNPy.Slider(NNPy.main.screen, sliderRect, sliderFunction)
			self.sliderList.append(newSlider)



		# Used to store test results
		#self.testOutput = ""


	def update(self, mouseState, dt):	

		for slider in self.sliderList:
			slider.update(mouseState)

		# Show the networks ability to recreate input with training data
		imageInput = NNPy.main.trainingData[self.testImage]
		imageOutput = NNPy.main.nn.getOutput(imageInput)
		self.imageRendererInput.setImageData(imageInput)
		self.imageRendererOutput.setImageData(imageOutput)

		# 
		sliderImage = NNPy.main.nn.getOutputStartingFromLayer(self.compressedInput, NNPy.main.nn.structure.argmin())
		self.imageRendererCustom.setImageData(sliderImage)

	def render(self, mouseState):

		#text = self.font.render("TESTING MODE :)", True, (0,0,0))
		#self.screen.blit(text, [760,105])

		#text = self.font.render("Results: " + self.testOutput, True, (0,0,0))
		#self.screen.blit(text, [760,155])


		self.imageRendererInput.render()
		self.imageRendererOutput.render()

		self.imageRendererCustom.render()

		for slider in self.sliderList:
			slider.render()


	def nextTestImage(self):
		self.testImage += 1
		self.testImage %= NNPy.main.trainingDataLength

	def prevTestImage(self):
		self.testImage -= 1
		self.testImage %= NNPy.main.trainingDataLength


	def useImage(self):
		# Uses the image currently viewing in the training data as the compressed tensor
		NNPy.main.nn.getOutput(NNPy.main.trainingData[self.testImage])
		compressedTensor = NNPy.main.nn.output[NNPy.main.nn.structure.argmin()].copy()

		for i in range(self.sliderCount):
			self.compressedInput[i] = compressedTensor[i]
			self.sliderList[i].slideValue = (compressedTensor[i]+1)/2

		#print(compressedTensor)







