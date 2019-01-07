
/*

	The neural network class
	Training process has been created modularly and split into multiple function

	Should probably scale dEdW based on number of weights feeding into individual neurons

*/

class NN {
	constructor(structure, LEARNING_RATE, network=undefined) {

		this.MIN_STARTING_WEIGHT = -0.25;
		this.MAX_STARTING_WEIGHT = 0.25;

		this.LEARNING_RATE = LEARNING_RATE;

		this.structure = structure;		// Array that represents neurons in each layer


		this.network = network;
		if (this.network == undefined) {	// If not using an already made network

			// Generates actual network 3D array
			this.network = [];	// Contains a list of layers
			for (let layerNum=0; layerNum<this.structure.length; layerNum++) {
				let finalLayer = (layerNum==this.structure.length-1);	// If creating final layer (output layer)
				let layer = [];		// Layers hold a list of neurons

				// Non-output layers have one more neuron due to bias
				let neuronCount = this.structure[layerNum] + !finalLayer;

				// 0 weights if final layer, otherwise have as many weights as there are neurons in the next layer
				let connectionCount = finalLayer ? 0 : this.structure[layerNum+1];

				for (let neuronNum=0; neuronNum<neuronCount; neuronNum++) {	
					let neuron = [];	// Neuron holds a list of weights for connections to next layer
					for (let weightNum=0; weightNum<connectionCount; weightNum++) {
						// Weight strength is a random value between the min and max starting weights
						let weight = mapValue(Math.random(), 0, 1, this.MIN_STARTING_WEIGHT, this.MAX_STARTING_WEIGHT);
						neuron.push(weight);
					}
					layer.push(neuron);
				}
				this.network.push(layer);
			}
		}

		//console.log(this.network)

		// Used to keep track of derivatives with respect to individual weights
		// Knowing dEdW means one can apply gradient descent to reduce error (E)
		// Allows for training batches as derivatives can be stored (and averaged) across data points
		this.dEdW = [];		
		for (let l=0; l<this.structure.length-1; l++) {		// Final layer has no weights
			this.dEdW.push([]);
			for (let n=0; n<this.structure[l]+1; n++) {		// bias dEdW's are needed in each layer
				// As many weights as neurons in next layer
				this.dEdW[l].push(new Array(this.structure[l+1])); 
				this.dEdW[l][n].fill(0)
			}
		}

		// Used to store derivative of error with respect to the input of each neuron
		// dEdW is calculated with dEdI
		// dEdI is also used to calculate dEdI of previous layer neurons
		// All layers are shifted down by one because first is not needed (layer 1 has index 0)
		this.dEdI = [];		
		for (let l=1; l<this.structure.length; l++) {		// dEdI required for all except first layer
			this.dEdI.push(new Array(this.structure[l]));	// dEdI is not needed for bias's
			this.dEdI[l-1].fill(0);
		}


		// Create the variable used to store netOutput
		// Initalising this variable beforehand makes calculating it
		this.netOutput = new Array(this.structure.length);
		for (let i=0; i<this.structure.length; i++) {
			this.netOutput[i] = [new Array(this.structure[i]), new Array(this.structure[i])];
			this.netOutput[i][0].fill(0);
			this.netOutput[i][1].fill(0);
		}



	}
	getOutput(input, finalOutput=false) {	// Returns output of network for a particular input
		// finalOutput==true returns simply the output of the network
		// finalOutput==false returns all neuron input/outputs pairs for training from all layers

		if (input.length != this.structure[0]) {
			console.log("Invalid Input length", "Expecting " + this.structure[0] + " but recieved " + input.length);
			return;
		}


		this.netOutput[0][1] = input;

		// For each layer (not first because already have input)
		for (let l=1; l<this.structure.length; l++) {
			// For some reason, defining all these variables before using them makes it like 3x faster
			let prevLayerOutput = this.netOutput[l-1][1];	// Output of neurons in previous later
			let prevLayerNeurons = this.network[l-1];		// Neurons of previous layer
			let currLaterInput = this.netOutput[l][0];		// Neuron input of current layer
			let currLaterOutput = this.netOutput[l][1];		// Neuron output of current layer

			// For each neuron in current layer (excluding bias)
			for (let n=0; n<this.structure[l]; n++) {
				// Reset sum from previous feedforward
				let sum = 0;

				// For each neuron in previous layer (excluding bias)
				for (let w=0; w<this.structure[l-1]; w++) {
					// Add weighted sum
					sum += prevLayerOutput[w] * prevLayerNeurons[w][n];
				}

				// Add bias to input sum
				sum += prevLayerNeurons[this.structure[l-1]][n];

				// Store sum
				currLaterInput[n] = sum;	

				// Apply activation function
				currLaterOutput[n] = Math.tanh(sum);
			}
		}

		if (finalOutput) {
			return this.netOutput[this.structure.length-1][1];
		} else {
			return this.netOutput;
		}


	}
	findGradDescent(netOutput, desired) {
		// Calculates the gradient of descent and adds to this.dEdW so 
		// that weight changes for entire batch can be calculated.

		// Function finds a local dEdI which is attribute of class to prevent recreating each time
		// Maybe make dEdI algorithmically global to function.
		// This could potentially provide improvements to learning in general
		// Probably also faster since dEdW only needs to be found once per batch instead of for each item.
		// Although dEdW is just dEdI * output running through it, so not much computation is saved

		// Find dEdI for last layer
		this.findOutputdEdI(netOutput, desired)

		// Find dEdI for each preceding layer starting from the 2nd last and not including the first
		this.finddEdI(netOutput);

		// Finds and adds to dEdW based on dEdI
		this.finddEdW(netOutput);

	}

	findOutputdEdI(netOutput, desired) {
		// Finds dEdI for output layer by comparing to desired output
		// Error is the difference squared -> E=(target - desired)^2

		let lastLayer = this.structure.length-1;
		for (var n=0; n<this.structure[this.structure.length-1]; n++) {
			// Derivative of difference squared
			this.dEdI[lastLayer-1][n] = 2 * (netOutput[lastLayer][1][n] - desired[n]);

			// Derivative of activation function
			this.dEdI[lastLayer-1][n] *= (1 - netOutput[lastLayer][1][n]*netOutput[lastLayer][1][n]);
		}
	}

	findInputdEdI(netOutput, setResult) {
		// Finds dEdI for the input layer of the network
		// Saves dEdI calculations to setResult array
		// Used for finding dEdI of final layer for generator networks

		for (var n=0; n<this.structure[0]; n++) {	// For each neuron in layer
			setResult[n] = 0;	// Reset value from previous train

			// For each dEdI in 2nd layer
			for (var w=0; w<this.structure[1]; w++) {		
				// Each neuron in next layer has a weighted effect on dEdI based on that neurons dEdI
				setResult[n] += this.network[0][n][w] * this.dEdI[0][w];	
			}

			// Activation function derivative
			setResult[n] *= (1 - netOutput[0][1][n]*netOutput[0][1][n]);
		}
	}

	finddEdI(netOutput) { 	// Finds dEdI for each layer excluding first and last
		// Starts from end and works its way backwards
		// Last layer is calculated another way and is assumed to already be set
		// First layer doesn't exist as it is not needed in finding dEdW

		// Start 2nd last layer and work backwards to 2nd layer
		for (var l=this.structure.length-1; l---1;) {
			for (var n=0; n<this.structure[l]; n++) {	// For each neuron in layer
				this.dEdI[l-1][n] = 0;	// Reset value from previous train

				// For each dEdI in next layer
				for (var w=0; w<this.structure[l+1]; w++) {		
					// Each neuron in next layer has a weighted effect on dEdI based on that neurons dEdI
					this.dEdI[l-1][n] += this.network[l][n][w] * this.dEdI[l][w];	
				}

				// Activation function derivative
				this.dEdI[l-1][n] *= (1 - netOutput[l][1][n]*netOutput[l][1][n]);
			}
		}
	}

	finddEdW(netOutput) {	// Finds dEdW based on current state of dEdI
		// Calculations are added to each value in this.dEdW rather than set to allow for batch training

		for (var l=0; l<this.structure.length-1; l++) {		// No weights in last layer
			// Find dEdW for normal neuron weights
			for (var n=0; n<this.structure[l]; n++) {		
				for (var w=0; w<this.structure[l+1]; w++) {
					this.dEdW[l][n][w] += netOutput[l][1][n] * this.dEdI[l][w];
				}
			}
			// Find dEdW for bias neuron weights
			for (var w=0; w<this.structure[l+1]; w++) {
				this.dEdW[l][this.structure[l]][w] += this.dEdI[l][w];
			}
		}

	}
	applydEdW() {	// Applies gradient descent using the current this.dEdW

		// Applies dEdW by altering each weight through gradient descent
		for (var l=0; l<this.structure.length-1; l++) {		// No weights in last layer
			for (var n=0; n<this.structure[l]+1; n++) {		// Include bias neuron weights
				for (var w=0; w<this.structure[l+1]; w++) {
					this.network[l][n][w] += -0.1 * this.LEARNING_RATE * this.dEdW[l][n][w];
				}
			}
		}

	}
	resetdEdW() {	// Resets dEdW to all zeros
		for (let l=0; l<this.structure.length-1; l++) {	
			for (let n=0; n<this.structure[l]+1; n++) {	
				this.dEdW[l][n].fill(0)
			}
		}
	}

	getError(output, desired) {
		let error = 0;
		for (let i=0; i<output.length; i++) {
			error += (output[i] - desired[i])**2;	// Squaring makes it positive and is good for gradient function
		}
		return error;
	}


	childNN(mutateRate=0) {		// Returns a mutated child network

		let mutateChance = 0.2;

		// Creates a slightly different child bot
		let childNetwork = [];
		for (let layerNum=0; layerNum<this.network.length; layerNum++) {
			let layer = [];
			for (let neuronNum=0; neuronNum<this.network[layerNum].length; neuronNum++) {	
				let neuron = [];
				for (let weightNum=0; weightNum<this.network[layerNum][neuronNum].length; weightNum++) {
					let weight = this.network[layerNum][neuronNum][weightNum];
					if (mutateRate > 0 && Math.random()<=mutateChance) {
						//weight += (2*Math.random() - 1) * mutateRate;
						weight += Math.log(1/Math.random() - 1) * mutateRate;
					}
					neuron.push(weight);
				}
				layer.push(neuron);
			}
			childNetwork.push(layer);
		}

		let newNN = new NN(this.structure, 0, childNetwork);

		return newNN;
	}
}
