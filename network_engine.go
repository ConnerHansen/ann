package main

// NetworkEngine the general interface for network engines
type NetworkEngine interface {
	Run(input [][]float64, network NetworkConfiguration) error
	Train(iterations int, config *TrainingConfiguration)
}
