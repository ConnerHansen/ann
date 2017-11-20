package main

import (
	"testing"
	"time"

	"github.com/connerhansen/this"
	. "github.com/onsi/gomega"
)

func init() {
	initLoggers()
	RegisterFailHandler(this.GomegaFailHandler)
}

func TestNeuronConnectionLogic(t *testing.T) {
	this.Should("Append a new incoming connection", t,
		func() {
			neuron := NewNeuron(TypeExcitatory)

			Expect(len(neuron.In)).To(Equal(0))
			neuron.AddIncoming(nil)
			Expect(len(neuron.In)).To(Equal(1))
		})

	this.Should("Append a new outgoing connection", t,
		func() {
			neuron := NewNeuron(TypeExcitatory)

			Expect(len(neuron.Out)).To(Equal(0))
			neuron.AddOutgoing(nil)
			Expect(len(neuron.Out)).To(Equal(1))
		})

	this.Should("Properly connect two neurons together", t,
		func() {
			neuronSrc := NewNeuron(TypeExcitatory)
			neuronTgt := NewNeuron(TypeExcitatory)

			Expect(len(neuronSrc.Out)).To(Equal(0))
			Expect(len(neuronTgt.In)).To(Equal(0))

			conn := neuronSrc.Connect(neuronTgt)
			conn.Weight = 1.0
			Expect(len(neuronSrc.Out)).To(Equal(1))
			Expect(len(neuronTgt.In)).To(Equal(1))

			Expect(neuronSrc.Out[0].Source).To(BeIdenticalTo(neuronSrc))
			Expect(neuronSrc.Out[0].Target).To(Equal(neuronTgt))
			Expect(neuronTgt.In[0].Target).To(Equal(neuronTgt))
			Expect(neuronSrc.Out[0].Weight).To(Equal(1.0))
		})
}

func TestNeuronEvents(t *testing.T) {
	this.Should("Return false and not fire when the neuron has no connections", t,
		func() {
			n1 := NewNeuron(TypeExcitatory)

			// Make sure the neuron is primed enough to actually fire
			n1.Potential = PotentialThreshold
			n1.Fire()
			Expect(n1.FiredAt).To(Equal(time.Time{}))
		})

	this.Should("Not reset the potential if the neuron doesn't fire", t,
		func() {
			n1 := NewNeuron(TypeExcitatory)

			// Make sure the neuron is not actually primed enough to fire
			n1.Potential = PotentialThreshold / 2
			Expect(n1.Fire()).To(BeFalse())
			Expect(n1.Potential).To(Equal(PotentialThreshold / 2))
		})

	this.Should("Update the timestamp after the neuron fires", t,
		func() {
			n1 := NewNeuron(TypeExcitatory)
			n2 := NewNeuron(TypeExcitatory)

			// Make sure the neuron is primed enough to fire
			n1.Potential = PotentialThreshold
			n1.Connect(n2)

			Expect(n1.FiredAt).To(Equal(time.Time{}))
			n1.Fire()
			Expect(n1.FiredAt).ToNot(Equal(time.Time{}))
		})

	this.Should("Reset the neuron's potential after it fires", t,
		func() {
			n1 := NewNeuron(TypeExcitatory)
			n2 := NewNeuron(TypeExcitatory)

			// Make sure the neuron is primed enough to fire
			n1.Potential = PotentialThreshold
			n1.Connect(n2)

			n1.Fire()
			Expect(n1.Potential).To(Equal(0.0))
		})

	this.Should("Calculate falloff based on the current potential and time since last fired", t,
		func() {
			this.Skip()
		})
}
