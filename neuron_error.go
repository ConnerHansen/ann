package main

import (
	"math"
)

// NeuronError the struct for storing the error associated with a specific
// neuron
type NeuronError struct {
	Direction   int
	Error       float64
	TotalWeight float64
}

// Sigmoid calculates error along a sigmoid curve
func (n *NeuronError) Sigmoid() float64 {
	return Sigmoid(n.Error)
}

// Sigmoid calculates error along a sigmoid curve
func Sigmoid(x float64) float64 {
	// e based sigmoid (0, 1)
	// return 1.0 / (1.0 + math.Pow(math.E, -x))

	// algebraic sigmoid (-1, 1)
	return x / math.Sqrt(1+math.Pow(x, 2))
}
