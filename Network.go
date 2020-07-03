package basicnn

import "math/rand"

// Network - the neural network
type Network struct {
	// learnRate - the learning rate of this neural network
	learnRate float64

	// layers - the hidden layer
	// consists of neurons in each layer
	layers [][]*Neuron

	// prediction - the output neuron
	prediction *Neuron
}

// NewNetworkInput - input for NewNetwork()
type NewNetworkInput struct {
	// Depth - how many layers in the neural network?
	Depth int

	// NeuronPerLayer - how many neurons in a layer?
	NeuronPerLayer int

	// LearningRate - the learning rate (0 - 1)
	LearningRate float64
}

// NewNetwork - creates a new neural network
func NewNetwork(input NewNetworkInput) Network {
	// initialize the hidden layers
	// 1. creates the layers based on `Depth`
	// 2. creates the neurons based on `NeuronPerLayer`
	layers := make([][]*Neuron, 0)
	for i := 1; i <= input.Depth; i++ { // 1
		neurons := make([]*Neuron, 0)
		for j := 1; j <= input.NeuronPerLayer; j++ { // 2
			neurons = append(neurons, NewNeuron(NewNeuronInput{
				Inputs:   nil,
				Weights:  nil,
				Bias:     rand.Float64(),
				IsOutput: false,
			}))
		}
		layers = append(layers, neurons)
	}

	// initialize the network
	network := Network{
		layers: layers,
		prediction: NewNeuron(NewNeuronInput{
			Inputs:   nil,
			Weights:  nil,
			Bias:     rand.Float64(),
			IsOutput: true,
		}),
		learnRate: input.LearningRate,
	}
	return network
}

// Train - trains a network [epoch] times based on a dataset []Data
func (n *Network) Train(epoch int, dataset []Data) {
	for i := 1; i <= epoch; i++ {
		for _, data := range dataset {
			n.singleEpoch(data)
		}
	}
}

// singleEpoch - a single epoch of training
// consists of:
//	1. FeedForward
//	2. Back propagation
//	3. Adjusting weights and bias
func (n *Network) singleEpoch(data Data) {
	// feed forward
	n.feedForward(data)

	// back propagation
	n.backPropagate()

	// update weights and biases
	n.adjustNeuronsWeightsAndBias(data)
}

// feedForward - performs a feed forward and get the final output from the output neuron
func (n *Network) feedForward(data Data) float64 {
	// feed forward
	for lIndex, layer := range n.layers {
		var inputs []float64
		if lIndex == 0 {
			// if this is the first layer in the hidden layer, initialize inputs with the variables from dataset
			inputs = data.Variables
		} else {
			// otherwise initialize input from the output of the previous layer
			inputs = make([]float64, 0)
			for _, neuron := range n.layers[lIndex-1] {
				inputs = append(inputs, neuron.Output)
			}
		}

		// initialize the inputs for each neuron and then perform a `feedforward` for each neuron
		for _, neuron := range layer {
			neuron.SetInputs(inputs)
			neuron.FeedForward()
		}

		// if it's the last layer, grab the output and feed it to the output neuron
		// then, perform a `feedforward` in the output neuron to get the final result
		if lIndex == len(n.layers)-1 {
			inputs = make([]float64, 0)
			for _, neuron := range layer {
				inputs = append(inputs, neuron.Output)
			}
			n.prediction.SetInputs(inputs)
			n.prediction.FeedForward()
		}
	}
	return n.prediction.Output
}

// backPropagate - performs a back propagation
// starts backwards (output layer -> first hidden layer)
func (n *Network) backPropagate() {
	// start with output layer
	n.prediction.CalculateAndStoreDerivatives()

	// start with layers in reverse
	for i := len(n.layers) - 1; i >= 0; i-- {
		layer := n.layers[i]
		for _, neuron := range layer {
			neuron.CalculateAndStoreDerivatives()
		}
	}
}

// adjustNeuronsWeightsAndBias - updates weights and bias values for each neuron
// starts backwards (output layer -> first hidden layer)
func (n *Network) adjustNeuronsWeightsAndBias(data Data) {
	// calculate the derivative of the final output
	derivativeY := n.deltaY(data)

	// start with output layer
	n.prediction.UpdateWeightsAndBias(n.learnRate, derivativeY, 0.0) // ignore the derivativeOutput argument as it's not used in the output neuron (third argument)

	// update the weights and bias in the reverse order in the hidden layer
	for i := len(n.layers) - 1; i >= 0; i-- {
		layer := n.layers[i]
		for j, neuron := range layer {
			var derivativeOutput float64
			if i == len(n.layers)-1 {
				// if the layer is directly before the output layer, grab the value from the output neuron.
				derivativeOutput = n.prediction.DerivativeInputs[j]
			} else {
				// get the mean of derivatives from all neurons from the next layer
				derivativeOutput = 0.0
				for _, neuron := range n.layers[i+1] {
					derivativeOutput += neuron.DerivativeInputs[j]
				}
				derivativeOutput /= float64(len(n.layers[i+1]))
			}

			// update the weights and bias for this neuron
			neuron.UpdateWeightsAndBias(n.learnRate, derivativeY, derivativeOutput)
		}
	}
}

// deltaY calculate derivative of error in respect of predictions[index]
// ---------------------------------------------------------------------------
// Etotal = (1 / number of output neurons) * (y_true - y_pred)^2
// ---------------------------------------------------------------------------
// d -> delta
// dEtotal -> delta Etotal
// dEtotal / dOutput
// -2 * (1 / number of output neurons) * (y_true - y_pred)
// this is called chain rule (https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-1a/v/chain-rule-introduction)
func (n Network) deltaY(data Data) float64 {
	return -2.0 * (data.Output - n.prediction.Output)
}

// Predict - predicts a data output based on the variables.
// it only calls feedForward(data)
func (n Network) Predict(data Data) float64 {
	return n.feedForward(data)
}
