package main

import "math"

var (

	// NeuronConnectionWeight the weight of a single neuron connection
	NeuronConnectionWeight = 0.1

	// NeuronConnectionCountStep the step size for how much to increment/decrement
	// the neuron connection count
	NeuronConnectionCountStep = 1

	// NeuronConnectionCountMinimum the minimum number of connections that the
	// neuron connection can have
	NeuronConnectionCountMinimum = 0

	// PotentialThreshold the minimum weight of a neuron connection before it will
	// fire against its outgoing connections
	PotentialThreshold = 0.0
)

// NeuronConnection the dendrites for the neurons
type NeuronConnection struct {
	Connections int     `json:"connections"`
	Source      *Neuron `json:"source"`
	Target      *Neuron `json:"target"`
	Weight      float64 `json:"weight"`
}

// NewNeuronConnection creates a new default neuron connection
func NewNeuronConnection(src, target *Neuron) *NeuronConnection {
	return &NeuronConnection{
		Connections: NeuronConnectionCountStep,
		Source:      src,
		Target:      target,
		Weight:      NeuronConnectionWeight,
	}
}

// CalculateIntensity calculates the final intensity of this neuron's fire
// event
func (n *NeuronConnection) CalculateIntensity() float64 {
	// Make sure excitatory neurons add to the potential and inhibitory neurons
	// subtract from it
	if n.Source.Type == TypeExcitatory {
		return float64(n.Connections) * n.Weight
	}

	return -1.0 * float64(n.Connections) * n.Weight
}

// Strengthen strengthens this connection by 1 step
func (n *NeuronConnection) Strengthen() int {
	n.Connections += NeuronConnectionCountStep
	return n.Connections
}

// Weaken weakens this connection by 1 step
func (n *NeuronConnection) Weaken() int {
	n.Connections -= NeuronConnectionCountStep

	if n.Connections < NeuronConnectionCountMinimum {
		n.Connections = NeuronConnectionCountMinimum
	}

	return n.Connections
}

// Fire fires the current dendrite connection from the source neuron to the
// target neuron
func (n *NeuronConnection) Fire() bool {
	if n.Source.Potential >= PotentialThreshold {
		n.Target.Potential += n.CalculateIntensity()

		if math.IsNaN(n.Target.Potential) || math.IsInf(n.Target.Potential, 1) || math.IsInf(n.Target.Potential, -1) {
			panic("fuck you")
		}

		return true
	}

	return false
}
