package basicnn

import "math"

// Sigmoid - sigmoid function
//	reference https://keisan.casio.com/exec/system/15157249643325
func Sigmoid(x float64) float64 {
	return float64(1) / (1 + (math.Exp(-x)))
}

// DerivedSigmoid - derivative of sigmoid
func DerivedSigmoid(sigmoid float64) float64 {
	return sigmoid * (1 - sigmoid)
}
