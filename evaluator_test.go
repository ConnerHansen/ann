package main

import (
	"math"
	"testing"

	"github.com/connerhansen/this"
	. "github.com/onsi/gomega"
)

func TestEvaluationBackPropagation(suite *testing.T) {

	this.Before(suite, func() {
		PotentialThreshold = math.Inf(-1.0)
	})

	this.Should("Alter the network weights using back propagation", suite,
		func() {
			this.Skip()
			network := NewNeuralNetwork(3, 2, 2)
			network.Debug = false
			expected := [][]float64{
				[]float64{1.0, 1.0},
				[]float64{1.0, 1.0},
			}

			// Store the network state from before the backpropagation event
			before := network.Clone()
			Evaluator.PerformBackPropagation(expected, network)
			for i, layer := range before.GetLayers() {
				// Don't bother comparing incoming weights for the input layer
				if i == 0 {
					continue
				}
				compLayer := network.GetLayers()[i]

				layer.EachNeuronWithIndex(func(n *Neuron, row int, column int) {
					compNeuron := compLayer.Neurons[row][column]
					Expect(n.TotalInput()).ToNot(Equal(compNeuron.TotalInput()))
				})
			}

		})

	this.Should("Show convergance with inhibitory neurons on a simple network (overfitting 500k)", suite,
		func() {
			this.Skip()
			// Turn off inhibitory neurons for simplicity
			InhibitoryNeuronDensity = 0.0
			network := NewNeuralNetwork(1, 3, 3)
			InhibitoryNeuronDensity = 0.2
			network.AddLayer(5, 5)
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
			errMap, _ := Evaluator.CalculateError(expected, network)
			errMapOrig := errMap
			// Train it
			Evaluator.Train(500000, config)

			// Now get the new error map
			errMap, _ = Evaluator.CalculateError(expected, network)

			// Ensure the error rate is pretty damned near zero
			output := network.GetOutput()
			totalError := 0.0
			for i := range expected {
				for j := range expected[i] {
					totalError += math.Abs(
						Evaluator.LinearError(expected[i][j], output.Neurons[i][j].Potential))
				}
			}
			Expect(totalError).ToNot(Equal(0.0))

			// We ran for a while, there should be less than 0.001 worth of error
			// in our network
			Expect(int(totalError * 1000.0)).To(Equal(0))

			for neuron := range errMap {
				Expect(errMapOrig[neuron].TotalWeight).ToNot(Equal(errMap[neuron].TotalWeight))
			}
		})

	this.Should("Show convergance on a simple network (overfitting)", suite,
		func() {
			this.Skip()
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
			errMap, _ := Evaluator.CalculateError(expected, network)
			errMapOrig := errMap
			// Train it
			Evaluator.Train(200000, config)

			// Now get the new error map
			errMap, _ = Evaluator.CalculateError(expected, network)

			// Ensure the error rate is pretty damned near zero
			output := network.GetOutput()
			totalError := 0.0
			for i := range expected {
				for j := range expected[i] {
					totalError += math.Abs(
						Evaluator.LinearError(expected[i][j], output.Neurons[i][j].Potential))
				}
			}
			Expect(totalError).ToNot(Equal(0.0))

			// We ran for a while, there should be less than 0.001 worth of error
			// in our network
			Expect(int(totalError * 1000.0)).To(Equal(0))

			for neuron := range errMap {
				Expect(errMapOrig[neuron].TotalWeight).ToNot(Equal(errMap[neuron].TotalWeight))
			}
		})

}

func TestEvaluatorError(suite *testing.T) {

	this.Should("Calculate the total weight on errors appropriately", suite,
		func() {
			network := NewNeuralNetwork(2, 3, 3)
			network.AddLayer(2, 2)

			layer := network.GetOutput()

			expected := [][]float64{
				[]float64{0.0, 0.0},
				[]float64{0.0, 0.0},
			}

			errMap, _ := Evaluator.CalculateError(expected, network)

			layer.EachNeuron(func(n *Neuron) {
				weight := 0.0
				for _, conn := range n.In {
					weight += conn.Weight
				}

				Expect(errMap[n].TotalWeight).To(Equal(weight))
			})
		})

	this.Should("Calculate zero error when output matches expected values", suite,
		func() {
			network := NewNeuralNetwork(1, 2, 2)

			layer := network.GetOutput()
			layer.Neurons[0][0].Potential = 1.0
			layer.Neurons[0][1].Potential = 1.0
			layer.Neurons[1][0].Potential = 0.0
			layer.Neurons[1][1].Potential = 0.0

			expected := [][]float64{
				[]float64{1.0, 1.0},
				[]float64{0.0, 0.0},
			}

			errMap, _ := Evaluator.CalculateError(expected, network)
			Expect(errMap[layer.Neurons[0][0]].Error).To(Equal(0.0))
			Expect(errMap[layer.Neurons[0][1]].Error).To(Equal(0.0))
			Expect(errMap[layer.Neurons[1][0]].Error).To(Equal(0.0))
			Expect(errMap[layer.Neurons[1][1]].Error).To(Equal(0.0))
		})

	this.Should("Return an error when the output layer size mismatches expected set", suite,
		func() {
			network := NewNeuralNetwork(1, 2, 2)

			expected := [][]float64{
				[]float64{0.0, 0.0},
				[]float64{0.0, 0.0},
				[]float64{0.0, 0.0},
			}

			errMap, err := Evaluator.CalculateError(expected, network)
			Expect(errMap).To(BeNil())
			Expect(err).To(Equal(ErrArraySizeMismatch))
		})

	this.Should("Calculate the appropriate error when output doesn't match expected values", suite,
		func() {
			network := NewNeuralNetwork(1, 2, 2)

			layer := network.GetOutput()
			layer.Neurons[0][0].Potential = 1.0
			layer.Neurons[0][1].Potential = 1.0
			layer.Neurons[1][0].Potential = 0.0
			layer.Neurons[1][1].Potential = 0.0

			expected := [][]float64{
				[]float64{0.0, 0.0},
				[]float64{1.0, 1.0},
			}

			errMap, _ := Evaluator.CalculateError(expected, network)
			Expect(errMap[layer.Neurons[0][0]].Error).To(Equal(0.5))
			Expect(errMap[layer.Neurons[0][1]].Error).To(Equal(0.5))
			Expect(errMap[layer.Neurons[1][0]].Error).To(Equal(0.5))
			Expect(errMap[layer.Neurons[1][1]].Error).To(Equal(0.5))
		})

	this.Should("Return the linear error between two floats", suite,
		func() {
			err := Evaluator.LinearError(0.5, 0.1)
			Expect(err).To(Equal(-0.4))

			err = Evaluator.LinearError(0.1, 0.5)
			Expect(err).To(Equal(0.4))
		})

	this.Should("Return the mean squared error between two floats", suite,
		func() {
			// We have to round the output because of float point errors from the
			// actual calculation
			mse := Evaluator.MeanSquaredError(0.5, 0.1)
			Expect(round(mse, 5)).To(Equal(round(0.08, 5)))

			mse = Evaluator.MeanSquaredError(0.1, 0.5)
			Expect(round(mse, 5)).To(Equal(round(0.08, 5)))
		})

}
