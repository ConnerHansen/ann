package main

import "math"

// round takes the input float and rounds it to the specified number of digits
// of precision
func round(src float64, digits int) float64 {
	mult := math.Pow(10.0, float64(digits))
	return float64(int32(src*mult)) / mult
}

func sigmoid(val float64) float64 {
	return 1.0 / sigmoidBase(val)
}

func sigmoidBase(val float64) float64 {
	return 1.0 + math.Exp(-val)
}

// TODO: remove Sigmoid function in favor of this one
// sigmoidP derivative of the logarithmic sigmoid function
func sigmoidP(val float64) float64 {
	return (1.0 - sigmoid(val)) / sigmoidBase(val)
}
