package main

import (
	"math"
	"math/rand"
	"testing"

	"github.com/connerhansen/this"
	. "github.com/onsi/gomega"
)

func TestNeuralNetworkConfiguration(t *testing.T) {
	genInput := func(val float64, width, height int) [][]float64 {
		inputs := make([][]float64, width)
		for i := range inputs {
			inputs[i] = make([]float64, height)

			for j := range inputs[i] {
				inputs[i][j] = val
			}
		}

		return inputs
	}

	genRandInput := func(width, height int) [][]float64 {
		inputs := genInput(0.0, width, height)
		for i := range inputs {
			for j := range inputs[i] {
				inputs[i][j] = rand.Float64()
			}
		}

		return inputs
	}

	this.After(t, func() {
		// Reset the potential threshold to its default
		PotentialThreshold = 0.0
	})

	this.Before(t, func() {
		InhibitoryNeuronDensity = 0.5
		PotentialThreshold = 0.0
	})

	this.Should("Allow for layers of custom size to be added", t,
		func() {
			network := NewNeuralNetwork(0, 0, 0)

			network.AddLayer(5, 3)
			network.AddLayer(2, 7)
			network.AddLayer(3, 2)

			layers := network.GetLayers()
			Expect(len(layers[0].Neurons)).To(Equal(5))
			Expect(len(layers[0].Neurons[0])).To(Equal(3))
			Expect(len(layers[1].Neurons)).To(Equal(2))
			Expect(len(layers[1].Neurons[0])).To(Equal(7))
			Expect(len(layers[2].Neurons)).To(Equal(3))
			Expect(len(layers[2].Neurons[0])).To(Equal(2))
		})

	this.Should("Create an appropriate set of network layers", t,
		func() {
			network := NewNeuralNetwork(3, 5, 5)
			Expect(len(network.Layers)).To(Equal(3))
		})

	this.Should("Create layers of equal dimension", t,
		func() {
			width := 7
			height := 3
			network := NewNeuralNetwork(3, width, height)

			network.EachLayer(func(layer *NetworkLayer) {
				Expect(len(layer.Neurons)).To(Equal(width))
				Expect(len(layer.Neurons[0])).To(Equal(height))
			})
		})

	this.Should("Clear each layer back to zero during a run", t,
		func() {
			network := NewNeuralNetwork(3, 5, 5)

			// Assign random potentials to all layers
			network.EachLayer(func(layer *NetworkLayer) {
				layer.EachNeuron(func(n *Neuron) {
					n.Potential = rand.Float64()
				})
			})

			// Run the network with a zeroed out input and a potential threshold to
			// keep anything from firing
			PotentialThreshold = 0.5
			network.Run(genInput(0.0, 5, 5))
			network.EachLayer(func(layer *NetworkLayer) {
				layer.EachNeuronWithIndex(func(n *Neuron, row, col int) {
					Expect(n.Potential).To(Equal(0.0))
				})
			})
		})

	this.Should("Return an error if test input size does not match input layer size", t,
		func() {
			network := NewNeuralNetwork(3, 5, 5)
			err := network.Run(genInput(0.0, 3, 3))

			Expect(err).To(Equal(ErrArraySizeMismatch))
		})

	this.Should("Properly percolate a signal down the network", t,
		func() {
			PotentialThreshold = math.MinInt64
			InhibitoryNeuronDensity = 0.4
			network := NewNeuralNetwork(3, 5, 5)

			// Run the network with a zeroed out input and a potential threshold to
			// keep anything from firing
			network.Run(genRandInput(5, 5))
			allZero := true

			network.GetOutput().EachNeuronWithIndex(func(n *Neuron, row, col int) {
				allZero = allZero && n.Potential == 0.0
			})

			Expect(allZero).To(BeFalse())
		})
}
