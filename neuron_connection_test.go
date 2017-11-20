package main

import (
	"math"
	"testing"

	"github.com/connerhansen/this"
	. "github.com/onsi/gomega"
)

func TestNeuronConnectionGroups(suite *testing.T) {
	this.Before(suite, func() {
		PotentialThreshold = math.Inf(-1.0)
	})

	this.Should("Match both calculated input and actual input from layer to layer", suite,
		func() {
			layer1 := NewNetworkLayer(3, 3)
			layer2 := NewNetworkLayer(3, 3)

			layer1.Connect(layer2)

			// Make sure we're zeroed out first
			layer2.EachNeuron(func(n *Neuron) {
				Expect(n.Potential).To(Equal(0.0))
			})

			// Manually fire the layer
			layer1.EachNeuron(func(n *Neuron) {
				n.Fire()
			})

			// Make sure each neuron has some real value
			layer2.EachNeuron(func(n *Neuron) {
				Expect(n.Potential).ToNot(Equal(math.NaN()))
				Expect(n.Potential).ToNot(Equal(0.0))
			})
		})
}

func TestNetworkConnectionEvents(suite *testing.T) {
	this.Should("Calculate the appropriate intensity for firing connections", suite,
		func() {
			src := NewNeuron(TypeExcitatory)
			tgt := NewNeuron(TypeExcitatory)

			conn := src.Connect(tgt)

			// Make sure we got the defaults
			Expect(conn.Weight).To(Equal(NeuronConnectionWeight))
			Expect(conn.CalculateIntensity()).To(Equal(conn.Weight))
		})

	this.Should("Calculate the appropriate intensity for firing inhibitory connections", suite,
		func() {
			src := NewNeuron(TypeInhibitory)
			tgt := NewNeuron(TypeExcitatory)

			conn := src.Connect(tgt)

			// Make sure we got the defaults
			Expect(conn.Weight).To(Equal(NeuronConnectionWeight))
			Expect(conn.CalculateIntensity()).To(Equal(-1.0 * conn.Weight))
		})

	this.Should("Return false and not fire when the source potential is too low", suite,
		func() {
			n1 := NewNeuron(TypeExcitatory)
			n2 := NewNeuron(TypeExcitatory)
			conn := NewNeuronConnection(n1, n2)

			PotentialThreshold = 0.5
			Expect(n1.Potential).To(Equal(0.0))
			Expect(conn.Fire()).To(BeFalse())
			Expect(n2.Potential).To(Equal(0.0))
		})

	this.Should("Fire when the source potential is high enough", suite,
		func() {
			n1 := NewNeuron(TypeExcitatory)
			n2 := NewNeuron(TypeExcitatory)
			conn := NewNeuronConnection(n1, n2)

			PotentialThreshold = 0.5
			n1.Potential = 0.5
			Expect(conn.Fire()).To(BeTrue())
			Expect(n2.Potential).To(Equal(sigmoid(conn.Weight)))
		})

	// this.Should("Properly distinguish between inhibitory and excitatory neurons", suite,
	// 	func() {
	// 		n1 := NewNeuron(TypeExcitatory)
	// 		n2 := NewNeuron(TypeInhibitory)
	// 		n3 := NewNeuron(TypeExcitatory)
	// 		conn1 := NewNeuronConnection(n1, n3)
	// 		conn2 := NewNeuronConnection(n2, n3)
	//
	// 		PotentialThreshold = 0.0
	// 		conn1.Fire()
	// 		Expect(n3.Potential).To(Equal(sigmoid(conn1.Weight)))
	//
	// 		conn2.Fire()
	// 		Expect(n3.Potential).To(Equal(0.0))
	// 	})
}
