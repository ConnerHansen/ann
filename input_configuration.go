package main

// InputConfiguration stores the necessary data to run training simulations
// on a network
type InputConfiguration struct {
	Expected [][]float64 `json:"expected"`
	Weight   float64     `json:"weight"`
	Values   [][]float64 `json:"values"`
}
