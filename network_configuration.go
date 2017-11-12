package main

// NetworkConfiguration hm...
type NetworkConfiguration interface {
	Clear()
	Clone() NetworkConfiguration
	EachLayer(do func(layer *NetworkLayer))
	GetDebug() bool
	GetInput() *NetworkLayer
	GetLayers() []*NetworkLayer
	GetOutput() *NetworkLayer
	Print()
	Run([][]float64) error
	SetDebug(debug bool)
}
