package main

import (
	"fmt"
	"io"
	"math/rand"
	"os"
)

var (
	// InhibitoryNeuronDensity the density with which to create inhibitory neurons
	InhibitoryNeuronDensity = 0.0
)

// NetworkLayer an individual, 2D layer of neurons
type NetworkLayer struct {
	Neurons [][]*Neuron `json:"neurons"`
}

// NewNetworkLayer creates a new network layer of the specified width and height
func NewNetworkLayer(width, height int) *NetworkLayer {
	layer := &NetworkLayer{Neurons: make([][]*Neuron, width)}

	for i := range layer.Neurons {
		layer.Neurons[i] = make([]*Neuron, height)

		for j := 0; j < len(layer.Neurons[i]); j++ {
			// Create the right density of inhibitory and excitatory neurons
			if rand.Float64() < InhibitoryNeuronDensity {
				layer.Neurons[i][j] = NewNeuron(TypeInhibitory)
			} else {
				layer.Neurons[i][j] = NewNeuron(TypeExcitatory)
			}
		}
	}

	return layer
}

// Clear clears the current layer's state back to 0.0
func (l *NetworkLayer) Clear() {
	l.EachNeuron(func(n *Neuron) {
		n.Potential = 0.0
	})
}

// EachNeuron perform some action on each neuron in this layer
func (l *NetworkLayer) EachNeuron(do func(n *Neuron)) {
	for _, row := range l.Neurons {
		for _, neuron := range row {
			do(neuron)
		}
	}
}

// EachNeuronWithIndex performs some function against every neuron in the
// network and includes its spacial data as well
func (l *NetworkLayer) EachNeuronWithIndex(do func(n *Neuron, row int, column int)) {
	for i, row := range l.Neurons {
		for j, neuron := range row {
			do(neuron, i, j)
		}
	}
}

// Connect connects a given layer to the next layer
func (l *NetworkLayer) Connect(target *NetworkLayer) {
	// Grab each neuron in the source layer
	l.EachNeuron(func(src *Neuron) {
		// And connect it to each neuron in the target layer
		target.EachNeuron(func(tgtNeuron *Neuron) {
			conn := src.Connect(tgtNeuron)
			conn.Weight = rand.Float64()
		})
	})
}

// Height returns the height of the current layer
func (l *NetworkLayer) Height() int {
	return len(l.Neurons[0])
}

// Print prints out the current layer using the field specified. Values are
// "potential", "total_in", "total_out"
func (l *NetworkLayer) Print(field string) {
	currRow := 0
	l.EachNeuronWithIndex(func(n *Neuron, row, col int) {
		if row != currRow {
			io.WriteString(os.Stdout, "\n")
			currRow = row
		}

		val := n.Potential
		switch field {
		case "total_in":
			val = 0.0
			for _, input := range n.In {
				val += input.Weight
			}
		case "total_out":
			val = 0.0
			for _, input := range n.Out {
				val += input.Weight
			}
		}

		io.WriteString(os.Stdout, fmt.Sprintf(" %.3f", val))
	})
}

// PrintErrorOffset is a rage function for when I have no idea what the fuck is happening
// there now stop complaining that I need to comment this
func (l *NetworkLayer) PrintErrorOffset(errMap map[*Neuron]*NeuronError) {
	for _, row := range l.Neurons {
		vals := ""
		for _, neuron := range row {
			vals = fmt.Sprintf("%s%.3f(%.3f)[%.3f]\t", vals, errMap[neuron].TotalWeight,
				errMap[neuron].Error, neuron.Potential)
		}

		Info.Println(vals)
	}
}

// Width returns the width of this current layer
func (l *NetworkLayer) Width() int {
	return len(l.Neurons)
}
