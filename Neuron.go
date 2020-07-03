package basicnn

import (
	"errors"
	"math/rand"
)

// Neuron - neuron representative in the neural network
type Neuron struct {
	// Inputs of each neuron
	//	1. The first hidden layer will have the inputs from dataset
	//	2. Subsequent layer (including output) will have the inputs from the previous layer's outputs
	Inputs []float64

	// Dynamic variables which will be adjusted in training
	Weights []float64
	Bias    float64

	// The output of this neuron after the activation function
	Output float64

	// IsOutput - identifies whether the neuron is in the output layer or not
	IsOutput bool

	// DerivativeValues - for back propagation
	DerivativeWeights []float64
	DerivativeInputs  []float64
	DerivativeBias    float64
}

// NewNeuronInput - input for NewNeuron()
type NewNeuronInput struct {
	Inputs   []float64
	Weights  []float64
	Bias     float64
	IsOutput bool
}

// NewNeuron - creates a new neuron
func NewNeuron(input NewNeuronInput) *Neuron {
	return &Neuron{
		Inputs:            input.Inputs,
		Weights:           input.Weights,
		Bias:              input.Bias,
		IsOutput:          input.IsOutput,
		DerivativeWeights: make([]float64, len(input.Weights)),
		DerivativeInputs:  make([]float64, len(input.Inputs)),
	}
}

// SetInputs - set inputs for a given neuron
func (n *Neuron) SetInputs(inputs []float64) {
	n.Inputs = inputs
	n.DerivativeInputs = make([]float64, len(inputs))

	if len(n.Weights) != len(inputs) {
		weights := make([]float64, 0)
		for i := 0; i < len(inputs); i++ {
			weights = append(weights, rand.Float64())
		}
		n.Weights = weights
		n.DerivativeWeights = make([]float64, len(weights))
	}
}

// FeedForward - function to feed the activation function to the next layer
func (n *Neuron) FeedForward() (float64, error) {
	if len(n.Inputs) != len(n.Weights) {
		return 0.0, errors.New("mismatched length: input and weight")
	}
	total := n.Bias
	for i, input := range n.Inputs {
		weight := n.Weights[i]
		total += weight * input
	}
	n.Output = Sigmoid(total)
	return n.Output, nil
}

// CalculateAndStoreDerivatives - calculates and store derivatives for back propagation
// 1. Weights
// 2. Inputs (used in previous layer)
// 3. Bias
func (n *Neuron) CalculateAndStoreDerivatives() {
	n.calculateAndStoreDerivativesWeights()
	n.calculateAndStoreDerivativesInputs()
	n.calculateAndStoreDerivativesBias()
}

// calculateAndStoreDerivativesWeights - calculates and store each weights' derivative for back propagation
// Formula: input * DerivedSigmoid(output)
func (n *Neuron) calculateAndStoreDerivativesWeights() {
	dOut := DerivedSigmoid(n.Output)
	for index := range n.Weights {
		if len(n.Inputs) > index {
			n.DerivativeWeights[index] = n.Inputs[index] * dOut
		}
	}
}

// calculateAndStoreDerivativesInputs - calculates and store each inputs' derivative for back propagation.
// the value will not be used by this neuron. It will be used by the neurons in the previous layer.
// Formula: weight * DerivedSigmoid(output)
func (n *Neuron) calculateAndStoreDerivativesInputs() {
	dOut := DerivedSigmoid(n.Output)
	for index := range n.Inputs {
		n.DerivativeInputs[index] = n.Weights[index] * dOut
	}
}

// calculateAndStoreDerivativesBias - calculates and store the derivative of the neuron's bias for back propagation
// Formula: DerivedSigmoid(output)
func (n *Neuron) calculateAndStoreDerivativesBias() {
	n.DerivativeBias = DerivedSigmoid(n.Output)
}

// UpdateWeightsAndBias - updates weights and bias for the current epoch based on the feedback from back propagation
// 1. Update Weights
// 2. Update Bias
func (n *Neuron) UpdateWeightsAndBias(learnRate float64, deltaY float64, deltaOutput float64) {
	n.updateWeights(learnRate, deltaY, deltaOutput)
	n.updateBias(learnRate, deltaY, deltaOutput)
}

// updateWeights - updates the weights based on the feedback from back propagation
// Formula for each weight:
//			current weight
//			- (learning rate
//			* derivativeY (from the final result)
//			* derivativeOutput (from derivative input if the neuron is not in the output layer. Otherwise this is `1`)
//			* derivativeWeight)
func (n *Neuron) updateWeights(learnRate float64, deltaY float64, deltaOutput float64) {
	for i := range n.Weights {
		n.Weights[i] -= n.calculateAdjustmentForWeightOrBias(learnRate, deltaY, deltaOutput, n.DerivativeWeights[i])
	}
}

// updateBias - updates the bias based on the feedback from back propagation
// Formula: current bias
//			- (learning rate
//			* derivativeY (from the final result)
//			* derivativeOutput (from derivative input if the neuron is not in the output layer. Otherwise this is `1`)
//			* derivativeWeight)
func (n *Neuron) updateBias(learnRate float64, deltaY float64, deltaOutput float64) {
	n.Bias -= n.calculateAdjustmentForWeightOrBias(learnRate, deltaY, deltaOutput, n.DerivativeBias)
}

// calculateAdjustmentForWeightOrBias - calculates the adjustment for bias or weight
// Formula: learning rate
//			* derivativeY (from the final result)
//			* derivativeOutput (from derivative input if the neuron is not in the output layer. Otherwise this is `1`)
//			* derivativeValue (weight OR bias)
func (n Neuron) calculateAdjustmentForWeightOrBias(learnRate float64, derivativeY float64, derivativeOutput float64, derivedValue float64) float64 {
	if n.IsOutput {
		return learnRate * derivativeY * derivedValue
	}
	return learnRate * derivativeY * derivativeOutput * derivedValue
}
