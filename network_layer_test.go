package main

import (
	"math/rand"
	"testing"
	"time"

	"github.com/connerhansen/this"
	. "github.com/onsi/gomega"
)

func TestNetworkLayerConnections(t *testing.T) {
	this.After(t, func() {
		InhibitoryNeuronDensity = 0.5
	})

	this.Should("Assign random weights to newly connected layers", t,
		func() {
			layer1 := NewNetworkLayer(5, 5)
			layer2 := NewNetworkLayer(5, 5)
			allZero := true

			layer1.Connect(layer2)
			layer1.EachNeuron(func(n *Neuron) {
				for _, conn := range n.Out {
					if conn.Weight != 0.0 {
						allZero = false
					}
				}
			})

			Expect(allZero).To(BeFalse())
		})

	this.Should("Distribute inhibitory neurons based on the threshold", t,
		func() {
			InhibitoryNeuronDensity = 0.5
			layer := NewNetworkLayer(10, 10)
			fuzzyWindow := 0.25 /* Allow some variation */
			inhibitoryCount := 0

			layer.EachNeuron(func(n *Neuron) {
				if n.Type == TypeInhibitory {
					inhibitoryCount++
				}
			})

			size := float64(10 * 10)
			ratio := float64(inhibitoryCount) / size
			min := size * (InhibitoryNeuronDensity - fuzzyWindow)
			max := size * (InhibitoryNeuronDensity + fuzzyWindow)

			Expect(ratio*size >= min && ratio*size <= max).To(BeTrue())
		})

	this.Should("Fully connect two layers together", t,
		func() {
			layer1 := NewNetworkLayer(5, 5)
			layer2 := NewNetworkLayer(10, 10)

			layer1.Connect(layer2)
			layer1.EachNeuron(func(n *Neuron) {
				Expect(len(n.Out)).To(Equal(100))
			})

			layer2.EachNeuron(func(n *Neuron) {
				Expect(len(n.In)).To(Equal(25))
			})
		})
}

func TestNetworkLayerConnectionConfiguration(t *testing.T) {
	connectLayerSet := func(layers []*NetworkLayer) {
		rand.Seed(time.Now().UnixNano())

		for i := 0; i < len(layers)-1; i++ {
			layer := layers[i]
			nextLayer := layers[i+1]

			// Connect the current layer to the next layer
			layer.Connect(nextLayer)
		}
	}

	this.Should("Reset the network to 0 on clear", t,
		func() {
			layer := NewNetworkLayer(5, 5)
			layer.EachNeuron(func(n *Neuron) {
				n.Potential = rand.Float64()
			})
			layer.Clear()

			layer.EachNeuron(func(n *Neuron) {
				Expect(n.Potential).To(Equal(0.0))
			})
		})

	this.Should("Fully connect all of the layers in a network", t,
		func() {
			layers := make([]*NetworkLayer, 5)
			layers[0] = NewNetworkLayer(2, 2)
			layers[1] = NewNetworkLayer(4, 4)
			layers[2] = NewNetworkLayer(6, 6)
			layers[3] = NewNetworkLayer(8, 8)
			layers[4] = NewNetworkLayer(10, 10)

			connectLayerSet(layers)

			for i, layer := range layers {
				if i > 0 {
					val := (2 * i) * (2 * i)
					Expect(len(layer.Neurons[0][0].In)).To(Equal(val))
				} else {
					Expect(len(layer.Neurons[0][0].In)).To(Equal(0))
				}

				if i < len(layers)-1 {
					val := (2 * (i + 2)) * (2 * (i + 2))
					Expect(len(layer.Neurons[0][0].Out)).To(Equal(val))
				} else {
					Expect(len(layer.Neurons[0][0].Out)).To(Equal(0))
				}
			}
		})
}
