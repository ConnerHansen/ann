package main

import "math"

var (

	// NeuronConnectionWeight the weight of a single neuron connection
	NeuronConnectionWeight = 0.1

	// NeuronConnectionCountMinimum the minimum number of connections that the
	// neuron connection can have
	NeuronConnectionCountMinimum = 0

	// PotentialThreshold the minimum weight of a neuron connection before it will
	// fire against its outgoing connections
	PotentialThreshold = 0.0
)

// NeuronConnection the dendrites for the neurons
type NeuronConnection struct {
	Source *Neuron `json:"source"`
	Target *Neuron `json:"target"`
	Weight float64 `json:"weight"`
}

// NewNeuronConnection creates a new default neuron connection
func NewNeuronConnection(src, target *Neuron) *NeuronConnection {
	return &NeuronConnection{
		Source: src,
		Target: target,
		Weight: NeuronConnectionWeight,
	}
}

// CalculateIntensity calculates the final intensity of this neuron's fire
// event
func (n *NeuronConnection) CalculateIntensity() float64 {
	// Make sure excitatory neurons add to the potential and inhibitory neurons
	// subtract from it
	if n.Source.Type == TypeExcitatory {
		return n.Weight
	}

	return -n.Weight
}

// Fire fires the current dendrite connection from the source neuron to the
// target neuron
func (n *NeuronConnection) Fire() bool {
	if n.Source.Potential >= PotentialThreshold {
		n.Target.Potential += sigmoid(n.Weight + n.Source.Bias)

		if math.IsNaN(n.Target.Potential) || math.IsInf(n.Target.Potential, 1) || math.IsInf(n.Target.Potential, -1) {
			panic("fuck you")
		}

		return true
	}

	return false
}

// Magnitude the positive value of the weight of this neuron
func (n *NeuronConnection) Magnitude() float64 {
	return math.Abs(n.Weight)
}
