package main

import (
	"math"
	"testing"

	"github.com/connerhansen/this"
	. "github.com/onsi/gomega"
)

func TestNetworkEngine(suite *testing.T) {

	buildComplicatedInput := func(network NetworkConfiguration) *TrainingConfiguration {
		input1 := [][]float64{
			[]float64{1.0, 1.0, 1.0},
			[]float64{1.0, 1.0, 1.0},
			[]float64{1.0, 1.0, 1.0},
		}

		expected1 := [][]float64{
			[]float64{1.0},
		}

		inputConfig1 := &InputConfiguration{
			Expected: expected1,
			Values:   input1,
			Weight:   1.0,
		}

		input2 := [][]float64{
			[]float64{0.0, 0.0, 0.0},
			[]float64{0.0, 0.0, 0.0},
			[]float64{0.0, 0.0, 0.0},
		}

		expected2 := [][]float64{
			[]float64{0.0},
		}

		inputConfig2 := &InputConfiguration{
			Expected: expected2,
			Values:   input2,
			Weight:   1.0,
		}

		return &TrainingConfiguration{
			Debug: true,
			Inputs: []*InputConfiguration{
				inputConfig1, inputConfig2,
			},
			Network: network,
		}
	}

	this.Should("Show convergance on a simple network (overfitting)", suite,
		func() {
			// Turn off inhibitory neurons for simplicity
			InhibitoryNeuronDensity = 0.0
			network := NewNeuralNetwork(1, 3, 3)
			network.AddLayer(3, 3)
			network.AddLayer(2, 2)

			input := [][]float64{
				[]float64{0.1, 0.9, 1.03},
				[]float64{0.51, 0.5, 0.5},
				[]float64{0.9, 0.85, 0.01},
			}

			expected := [][]float64{
				[]float64{0.25, 0.5},
				[]float64{0.75, 1.0},
			}

			config := &TrainingConfiguration{
				Debug: false,
				Inputs: []*InputConfiguration{
					&InputConfiguration{
						Expected: expected,
						Values:   input,
						Weight:   1.0,
					},
				},
				Network: network,
			}

			// Get the initial error mapping so we can make sure our error rate has
			// changed
			engine := &BackPropagationEngine{LearningRate: 0.1}

			// errMap := engine.BuildInputMap(expected, network)
			// errMap, _ := Evaluator.CalculateError(expected, network)
			// errMapOrig := errMap
			// Train it
			// Evaluator.Train(200000, config)
			engine.Train(20000, config)

			// Now get the new error map
			// errMap = engine.BuildInputMap(expected, network)

			// Ensure the error rate is pretty damned near zero
			output := network.GetOutput()
			totalError := 0.0
			for i := range expected {
				for j := range expected[i] {
					totalError += math.Abs(
						linearError(expected[i][j], output.Neurons[i][j].Potential))
				}
			}
			Expect(totalError).ToNot(Equal(0.0))

			// We ran for a while, there should be less than 0.001 worth of error
			// in our network
			Expect(int(totalError * 1000.0)).To(Equal(0))

			// for neuron := range errMap {
			// 	Expect(errMapOrig[neuron].TotalWeight).ToNot(Equal(errMap[neuron].TotalWeight))
			// }
		})

	this.Should("Show convergance on a more complex network (overfitting)", suite,
		func() {
			// this.Skip()
			// Turn off inhibitory neurons for simplicity
			InhibitoryNeuronDensity = 0.0
			network := NewNeuralNetwork(1, 3, 3)
			network.AddLayer(3, 3)
			network.AddLayer(3, 3)
			network.AddLayer(1, 1)

			config := buildComplicatedInput(network)
			input1 := config.Inputs[0]

			// Get the initial error mapping so we can make sure our error rate has
			// changed
			engine := &BackPropagationEngine{LearningRate: 0.01}

			// Train it
			engine.Train(200000, config)

			// Ensure the error rate is pretty damned near zero
			network.Run(input1.Values)
			output := network.GetOutput()
			totalError := 0.0
			for i := range input1.Expected {
				for j := range input1.Expected[i] {
					totalError += math.Abs(
						linearError(input1.Expected[i][j], output.Neurons[i][j].Potential))
				}
			}
			Expect(totalError).ToNot(Equal(0.0))

			// We ran for a while, there should be less than 0.001 worth of error
			// in our network
			Expect(int(totalError * 1000.0)).To(Equal(0))

			// for neuron := range errMap {
			// 	Expect(errMapOrig[neuron].TotalWeight).ToNot(Equal(errMap[neuron].TotalWeight))
			// }
		})
}
