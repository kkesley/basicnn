package basicnn_test

import (
	"testing"

	"github.com/kkesley/basicnn"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewNeuron(t *testing.T) {
	input := basicnn.NewNeuronInput{
		Inputs:   []float64{1.0, 2.0, 1.1},
		Weights:  []float64{0.2, 0.3, 0.4},
		Bias:     2.22,
		IsOutput: true,
	}
	neuron := basicnn.NewNeuron(input)
	assert.Equal(t, input.Inputs, neuron.Inputs, "they should be equal")
	assert.Equal(t, input.Weights, neuron.Weights, "they should be equal")
	assert.Equal(t, input.Bias, neuron.Bias, "they should be equal")
	assert.Equal(t, input.IsOutput, neuron.IsOutput, "they should be equal")

}

func TestSetInputsChangeNeuronInputs(t *testing.T) {
	neuron := basicnn.NewNeuron(basicnn.NewNeuronInput{
		nil, []float64{2.0, 3.0}, 4.0, false,
	})
	inputs := []float64{2.0, 3.0}
	neuron.SetInputs(inputs)
	assert.Equal(t, inputs, neuron.Inputs, "they should be the same")
}

func TestFeedForwardSupportValidInput(t *testing.T) {
	neuron := basicnn.NewNeuron(basicnn.NewNeuronInput{
		nil, []float64{2.0, 3.0}, 4.0, false,
	})
	neuron.SetInputs([]float64{2.0, 3.0})
	output, err := neuron.FeedForward()
	require.Nil(t, err, "should not throw an error")

	expectedValue := basicnn.Sigmoid(17.0)
	assert.Equal(t, expectedValue, output, "they should be the same")
}

func TestFeedForwardCatchMismatchedInputLength(t *testing.T) {
	neuron := basicnn.NewNeuron(basicnn.NewNeuronInput{
		nil, []float64{2.0, 3.0}, 4.0, false,
	})
	_, err := neuron.FeedForward()
	require.NotNil(t, err, "should return error")
}

func TestCalculateAndStoreDerivatives(t *testing.T) {
	neuron := basicnn.Neuron{
		Inputs:   []float64{2.0, 1.0, 3.0},
		Weights:  []float64{1.0, 2.0, 3.0},
		Bias:     2.0,
		Output:   2.0,
		IsOutput: false,

		// DerivativeValues - for back propagation
		DerivativeWeights: make([]float64, 3),
		DerivativeInputs:  make([]float64, 3),
		DerivativeBias:    0.0,
	}
	dOut := basicnn.DerivedSigmoid(neuron.Output)
	neuron.CalculateAndStoreDerivatives()
	for i, dWeight := range neuron.DerivativeWeights {
		assert.Equal(t, dOut*neuron.Inputs[i], dWeight, "they should be the same")
	}
	for i, dInput := range neuron.DerivativeInputs {
		assert.Equal(t, dOut*neuron.Weights[i], dInput, "they should be the same")
	}
	assert.Equal(t, dOut, neuron.DerivativeBias, "they should be the same")
}

func TestUpdateWeightsAndBias(t *testing.T) {}
