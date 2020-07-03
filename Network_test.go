package basicnn_test

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/kkesley/basicnn"
)

func TestNetwork(t *testing.T) {
	rand.Seed(time.Now().UTC().UnixNano())
	network := basicnn.NewNetwork(basicnn.NewNetworkInput{
		Depth:          2,
		NeuronPerLayer: 2,
		LearningRate:   0.1,
	})
	data := []basicnn.Data{
		{
			Variables: []float64{-2.0, -1.0},
			Output:    1.0,
		},
		{
			Variables: []float64{25.0, 6.0},
			Output:    0.0,
		},
		{
			Variables: []float64{17.0, 4.0},
			Output:    0.0,
		},
		{
			Variables: []float64{-15.0, -6.0},
			Output:    1.0,
		},
	}
	network.Train(1000, data)

	emily := basicnn.Data{
		Variables: []float64{-7.0, -3.0},
	}
	frank := basicnn.Data{
		Variables: []float64{20.0, 2.0},
	}
	fmt.Printf("Emily: %f\n", network.Predict(emily))
	fmt.Printf("Frank: %f\n", network.Predict(frank))
}
