package main

import "math/rand"

// TrainingConfiguration the setup for running training simulations
type TrainingConfiguration struct {
	Debug   bool `json:"debug"`
	Engine  NetworkEngine
	Inputs  []*InputConfiguration `json:"inputs"`
	Network NetworkConfiguration  `json:"network"`
}

// PickInput picks a random input from the training set based on their given
// proportional weight
func (t *TrainingConfiguration) PickInput() *InputConfiguration {
	pick := rand.Float64()
	currWeight := 0.0

	for i, input := range t.Inputs {
		currWeight += input.Weight
		// If this is the last input or if we're past the pick cutoff, then choose
		// this input
		if i == len(t.Inputs)-1 || currWeight/t.TotalWeight() > pick {
			return input
		}
	}

	return t.Inputs[len(t.Inputs)-1]
}

// TotalWeight returns the total weight associated with this training set so
// that we scale our selection proportionally
func (t *TrainingConfiguration) TotalWeight() float64 {
	weight := 0.0
	for _, input := range t.Inputs {
		weight += input.Weight
	}

	return weight
}
