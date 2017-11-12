package main

import (
	"errors"
	"io"
	"os"
)

var (
	// ErrArraySizeMismatch is the error for when the input dimensions and the neural net's
	// initial layer are not the same size
	ErrArraySizeMismatch = errors.New("Input dimensions do not match the network interface")
)

// NeuralNetwork the struct for storing binary decision networks
type NeuralNetwork struct {
	Debug              bool            `json:"debug"`
	CurrentTimeStep    float64         `json:"current_time_step"`
	Layers             []*NetworkLayer `json:"layers"`
	PotentialStep      float64         `json:"potential_step"`
	PotentialThreshold float64         `json:"potential_threshold"`
	TimeStepSize       float64         `json:"time_step_size"`
}

// NewNeuralNetwork creates a new binary decision network
func NewNeuralNetwork(depth, width, height int) *NeuralNetwork {
	layers := make([]*NetworkLayer, 0)

	b := &NeuralNetwork{
		Debug:              false,
		CurrentTimeStep:    0.0,
		Layers:             layers,
		PotentialStep:      1.0,
		PotentialThreshold: 0.0, /* Always fire -- continuous neurons */
		TimeStepSize:       1.0,
	}

	// Handle bad depth values properly -- just treat them like zero
	for i := 0; i < depth; i++ {
		b.AddLayer(width, height)
	}

	return b
}

// AddLayer adds a new layer to this network with the specified width and
// height. This will then connect the new layer with the previous layer of the
// network if a previous layer exists
func (n *NeuralNetwork) AddLayer(width, height int) {
	var currTail *NetworkLayer
	if len(n.Layers) > 0 {
		currTail = n.GetOutput()
	}
	newTail := NewNetworkLayer(width, height)
	n.Layers = append(n.Layers, newTail)

	if currTail != nil {
		// Wire 'em up
		currTail.Connect(newTail)
	}
}

// Clear resets the entire network back to 0 potential
func (n *NeuralNetwork) Clear() {
	for _, layer := range n.Layers {
		layer.Clear()
	}
}

// Clone replicates the binary network
func (n *NeuralNetwork) Clone() NetworkConfiguration {
	clone := NewNeuralNetwork(0, 0, 0)
	cloneMap := make(map[*Neuron]*Neuron)

	n.EachLayer(func(layer *NetworkLayer) {
		cloneLayer := NewNetworkLayer(layer.Width(), layer.Height())
		clone.Layers = append(clone.Layers, cloneLayer)

		// Clone the current layer, and track the source neuron to clone neuron
		// mapping so we can rebuild the connections
		layer.EachNeuronWithIndex(func(n *Neuron, row int, column int) {
			cloneNeuron := n.Clone()
			cloneLayer.Neurons[row][column] = cloneNeuron
			cloneMap[n] = cloneNeuron

			// Walk the incoming connections so each new layer properly wires itself
			// up to the previous layer
			for _, conn := range n.In {
				cloneSrc := cloneMap[conn.Source]

				cloneConn := cloneSrc.Connect(cloneNeuron)
				cloneConn.Connections = conn.Connections
				cloneConn.Weight = conn.Weight
			}
		})
	})

	return clone
}

// GetDepth returns the depth of the network
func (n *NeuralNetwork) GetDepth() int {
	return len(n.Layers)
}

// EachLayer convenience function for doing something on each layer
func (n *NeuralNetwork) EachLayer(do func(layer *NetworkLayer)) {
	for _, layer := range n.Layers {
		do(layer)
	}
}

// GetDebug return debug
func (n *NeuralNetwork) GetDebug() bool {
	return n.Debug
}

// GetInput returns the input layer of the network
func (n *NeuralNetwork) GetInput() *NetworkLayer {
	return n.Layers[0]
}

// GetLayers returns the layers of the network
func (n *NeuralNetwork) GetLayers() []*NetworkLayer {
	return n.Layers
}

// GetOutput returns the output layer of the network
func (n *NeuralNetwork) GetOutput() *NetworkLayer {
	return n.Layers[len(n.Layers)-1]
}

// Print prints out the current network's potential values
func (n *NeuralNetwork) Print() {
	// if n.Debug {
	io.WriteString(os.Stdout, "\n-----------------------------\n")

	n.EachLayer(func(layer *NetworkLayer) {
		io.WriteString(os.Stdout, "\n")
		layer.Print("")
		io.WriteString(os.Stdout, "\n")
	})

	io.WriteString(os.Stdout, "\n-----------------------------\n")
	// }
}

// Run processes the current neural net with the provided set of inputs. Since
// this is a binary neural net, the final solution is simply a true or false
func (n *NeuralNetwork) Run(inputs [][]float64) error {
	inputLayer := n.GetInput()
	if len(inputs) != len(inputLayer.Neurons) ||
		len(inputs[0]) != len(inputLayer.Neurons[0]) {
		return ErrArraySizeMismatch
	}

	// Reset the network before each run
	n.Clear()

	inputLayer.EachNeuronWithIndex(func(n *Neuron, row, column int) {
		n.Potential = inputs[row][column]
	})

	// Skip the output layer, since we don't want to try to fire that
	if n.Debug {
		Debug.Println("Starting run")
	}
	// n.Print("")
	for i := 0; i < len(n.Layers)-1; i++ {
		layer := n.Layers[i]

		// TODO: clean this up holy shit
		if n.Debug {
			io.WriteString(os.Stdout, "\n")
			layer.Print("")
			io.WriteString(os.Stdout, "\n")
		}

		// Try to fire every neuron in the layer
		layer.EachNeuron(func(neuron *Neuron) {
			neuron.Fire()
		})
	}

	if n.Debug {
		io.WriteString(os.Stdout, "\n")
		n.GetOutput().Print("")
		io.WriteString(os.Stdout, "\n")
	}

	n.CurrentTimeStep += n.TimeStepSize
	return nil
}

// SetDebug sets the debug level
func (n *NeuralNetwork) SetDebug(debug bool) {
	n.Debug = debug
}
