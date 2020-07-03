package basicnn_test

import (
	"testing"

	"github.com/kkesley/basicnn"
)

func TestSigmoid(t *testing.T) {
	// testCases - [input, result]
	testCases := [][2]float64{
		{7.0, 0.99908894880559935},
		{-2.0, 0.11920292202211755},
	}
	for _, testCase := range testCases {
		if result := basicnn.Sigmoid(testCase[0]); result != testCase[1] {
			t.Fatalf("Invalid result for case [%g]. Expected result of [%g] - Actual [%g]", testCase[0], testCase[1], result)
		}
	}
}
