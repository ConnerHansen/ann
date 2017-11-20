package main

import (
	"math"
	"testing"

	"github.com/connerhansen/this"
	. "github.com/onsi/gomega"
)

func TestComplicatedTraining(suite *testing.T) {

	this.Should("Converge on a solution that can successfully generalize", suite,
		func() {
			this.Skip()
			// Turn off inhibitory neurons for simplicity
			InhibitoryNeuronDensity = 0.0
			network := NewNeuralNetwork(1, 5, 5)
			network.AddLayer(5, 5)
			network.AddLayer(1, 1)

			config := &TrainingConfiguration{
				Debug: true,
				Inputs: []*InputConfiguration{
					&InputConfiguration{
						Expected: [][]float64{
							[]float64{1.0},
						},
						Values: [][]float64{
							[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
							[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
							[]float64{1.0, 1.0, 0.0, 0.0, 1.0},
							[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
							[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
						},
						Weight: 1.0},
					// &InputConfiguration{
					// 	Expected: [][]float64{
					// 		[]float64{1.0},
					// 		[]float64{1.0},
					// 	},
					// 	Values: [][]float64{
					// 		[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
					// 		[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
					// 		[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
					// 		[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
					// 		[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
					// 	},
					// 	Weight: 1.0},
					// &InputConfiguration{
					// 	Expected: [][]float64{
					// 		[]float64{0.0},
					// 	},
					// 	Values: [][]float64{
					// 		[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
					// 		[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
					// 		[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
					// 		[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
					// 		[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
					// 	},
					// 	Weight: 1.0},
					// &InputConfiguration{
					// 	Expected: [][]float64{
					// 		[]float64{0.0},
					// 		[]float64{1.0},
					// 	},
					// 	Values: [][]float64{
					// 		[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
					// 		[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
					// 		[]float64{1.0, 1.0, 0.0, 0.0, 1.0},
					// 		[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
					// 		[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
					// 	},
					// 	Weight: 1.0},
					// &InputConfiguration{
					// 	Expected: [][]float64{
					// 		[]float64{1.0},
					// 		[]float64{0.0},
					// 	},
					// 	Values: [][]float64{
					// 		[]float64{1.0, 0.7, 1.4, 0.74, 1.1},
					// 		[]float64{1.0, 1.0, 1.2, 0.5, 0.9},
					// 		[]float64{1.2, 0.455, 0.8, 0.1, 0.2},
					// 		[]float64{0.1, 0.0, 0.221, 0.5, 0.0},
					// 		[]float64{0.0, 0.5, 0.0, 0.0, 0.2},
					// 	},
					// 	Weight: 1.0},
					// &InputConfiguration{
					// 	Expected: [][]float64{
					// 		[]float64{0.0},
					// 		[]float64{1.0},
					// 	},
					// 	Values: [][]float64{
					// 		[]float64{0.0, 0.5, 0.0, 1.0, 0.2},
					// 		[]float64{0.1, 0.2, 0.0, 0.5, 0.0},
					// 		[]float64{1.2, 0.2, 0.8, 1.1, 0.2},
					// 		[]float64{1.0, 0.7, 1.4, 0.74, 1.1},
					// 		[]float64{0.8, 0.4, 1.2, 0.5, 0.9},
					// 	},
					// 	Weight: 1.0},
					// &InputConfiguration{
					// 	Expected: [][]float64{
					// 		[]float64{1.0},
					// 		[]float64{0.0},
					// 	},
					// 	Values: [][]float64{
					// 		[]float64{1.0, 0.2, 0.9, 1.0, 0.2},
					// 		[]float64{1.0, 0.1, 1.2, 1.0, 0.9},
					// 		[]float64{1.2, 1.0, 0.8, 0.1, 0.9},
					// 		[]float64{0.1, 0.533, 0.221, 0.023, 0.433},
					// 		[]float64{0.4, 0.5, 0.022, 0.0, 0.2},
					// 	},
					// 	Weight: 1.0},
					// &InputConfiguration{
					// 	Expected: [][]float64{
					// 		[]float64{0.0},
					// 		[]float64{1.0},
					// 	},
					// 	Values: [][]float64{
					// 		[]float64{0.644, 0.023, 0.0, 0.753, 0.2},
					// 		[]float64{0.011, 0.1, 1.0, 0.01, 0.022},
					// 		[]float64{0.0, 1.0, 0.2, 0.01, 0.2},
					// 		[]float64{1.0, 0.7, 1.01, 0.1344, 1.1},
					// 		[]float64{1.0, 0.4, 0.644, 0.21, 0.732},
					// 	},
					// 	Weight: 1.0},
				},
				Network: network,
			}

			// Train it
			Evaluator.Train(50000, config)

			// Obvious test
			test := [][]float64{
				[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
				[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
				[]float64{0.5, 0.5, 0.5, 0.5, 0.5},
				[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
				[]float64{0.0, 0.0, 0.0, 0.0, 0.0},
			}

			expected := [][]float64{
				[]float64{1.0},
			}

			// Now run our actual test input without back propagation
			network.Run(test)
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
			Debug.Println("Total error:", totalError)
		})

}
