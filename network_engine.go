package main

import "math"

// NetworkEngine the general interface for network engines
type NetworkEngine interface {
	Train(iterations int, config *TrainingConfiguration)
}

// BackPropagationEngine the normal engine for running a standard back
// propagation algorithm
type BackPropagationEngine struct {
	LearningRate float64 `json:"learning_rate"`
}

// BackPropagateLayer performs the back propagation algorithm across a given
// layer
func (b *BackPropagationEngine) BackPropagateLayer(layer *NetworkLayer,
	vals map[*Neuron]float64) map[*Neuron]float64 {
	newVals := make(map[*Neuron]float64)

	layer.EachNeuron(func(n *Neuron) {
		for _, conn := range n.In {
			if val, ok := vals[n]; ok {
				delta := b.WeightDelta(val, conn)
				conn.Weight += delta

				// Now track how much accumulated delta we have on each source neuron
				newVals[conn.Source] += delta
				// Debug.Println("Delta:", delta)
			} else {
				Error.Println("fucking fucking fuck what the fuck")
			}
		}
	})

	// Normalize the aggregated weights
	// layer.EachNeuron(func(n *Neuron) {
	// 	for _, conn := range n.In {
	// 		conn.Weight = sigmoid(conn.Weight)
	// 	}
	// })

	// Normalize the error
	// for neuron, val := range newVals {
	// 	newVals[neuron] = sigmoid(val)
	// }

	return newVals
}

func (b *BackPropagationEngine) BuildInputMap(expected [][]float64,
	network NetworkConfiguration) map[*Neuron]float64 {
	vals := make(map[*Neuron]float64)
	output := network.GetOutput()

	output.EachNeuronWithIndex(func(n *Neuron, row int, column int) {
		vals[n] = expected[row][column]
	})

	return vals
}

// // AdjustLayer performs the actual fine tuning of the current layer given a base
// // error mapping. This returns the error mapping for the current layer
// func (b *BackPropagationEngine) AdjustLayer(layer *NetworkLayer, errMap map[*Neuron]*NeuronError) (map[*Neuron]*NeuronError, error) {
// 	currErrMap := make(map[*Neuron]*NeuronError)
//
// 	// Go through each neuron in the current layer and adjust the outgoing
// 	// weights accordingly. Track how much adjustment it takes overall
// 	layer.EachNeuron(func(n *Neuron) {
// 		currErrMap[n] = &NeuronError{Direction: 1}
//
// 		// Get the total weight coming in to this neuron and stored that for future
// 		// usage
// 		for _, conn := range n.In {
// 			currErrMap[n].TotalWeight += conn.Weight
// 		}
//
// 		// Now, figure out how much to adjust the incoming weights
// 		for _, conn := range n.Out {
// 			err := errMap[conn.Target]
//
// 			// How much of the error is this connection responsible for?
// 			proportionalWeight := conn.Weight / conn.Target.TotalInputWeight()
//
// 			// Now, it's proportional, but we need to adjust from gross to fine tuning
// 			// and taking our current weight into account alongside the sigmoid helps.
// 			// Right now we've got the scaling value cranked all the way up because it
// 			// makes things converge more rapidly
// 			adjStep := 0.1 * proportionalWeight * err.Sigmoid()
// 			// adjStep := 0.2 * conn.Weight * proportionalWeight * err.Sigmoid()
//
// 			// Keep track of how much error we have at this layer so we can percolate
// 			// that up
// 			currErrMap[n].Error += adjStep
//
// 			// Now do the actual adjustment
// 			conn.Weight += adjStep * float64(err.Direction)
// 		}
//
// 		// Make sure we keep our logic consistent (ie separate magnitude and direction)
// 		if currErrMap[n].Error < 0 {
// 			currErrMap[n].Direction = -1
// 			currErrMap[n].Error *= -1
// 		}
//
// 		// Make sure we scale the error appropriately
// 		// currErrMap[n].Error = e.MeanSquaredError(0.0, currErrMap[n].Error)
// 	})
//
// 	return currErrMap, nil
// }

// // CalculateNetworkError calculates the error of the output layer versuses the provided
// // set of expected values
// func CalculateNetworkError(expected [][]float64,
// 	network NetworkConfiguration) (map[*Neuron]*NeuronError, error) {
// 	layer := network.GetOutput()
//
// 	if len(expected) != len(layer.Neurons) ||
// 		len(expected[0]) != len(layer.Neurons[0]) {
// 		return nil, ErrArraySizeMismatch
// 	}
//
// 	errMap := make(map[*Neuron]*NeuronError)
//
// 	for i, row := range expected {
// 		for j, val := range row {
// 			weight := 0.0
// 			neuron := layer.Neurons[i][j]
// 			for _, input := range neuron.In {
// 				weight += input.Weight
// 			}
//
// 			direction := 1
// 			if val < neuron.Potential {
// 				direction = -1
// 			}
//
// 			err := &NeuronError{
// 				Direction:   direction,
// 				Error:       meanSquaredError(val, neuron.Potential),
// 				TotalWeight: weight,
// 			}
// 			errMap[neuron] = err
// 		}
// 	}
//
// }

// WeightDelta calculates the delta to apply to a given weight on a neuron connection
func (b *BackPropagationEngine) WeightDelta(expected float64, conn *NeuronConnection) float64 {
	// potential := conn.Target.Potential
	// Debug.Printf("%.3f %.3f %.3f %.3f\n", expected, potential, linearError(expected, conn.Target.Output()), conn.Weight)

	// Delta Rule: learning rate * (target output - actual output) g'(sum of weights) * x
	// return b.LearningRate * linearError(expected, sigmoid(potential)) * sigmoidP(potential) *
	// (conn.Weight + bias)
	// Simplified with linear activation function:
	// learning rate * (linear error of expected vs actual output) * weight of this connection
	return b.LearningRate * linearError(expected, conn.Target.Potential) * (sigmoid(conn.Weight + conn.Source.Bias))
}

// Train runs the training configuration a certain number of iterations
func (b *BackPropagationEngine) Train(iterations int, config *TrainingConfiguration) {
	network := config.Network

	// Log 100 frames if we're debugging
	debugLogTick := iterations / 100
	input := config.PickInput()

	// Only switch inputs 1000 times -- this is a hardcoded limit for now
	// trainingBlockCount := iterations / 1000
	trainingBlockCount := 100
	for i := 0; i <= iterations; i++ {

		if i%trainingBlockCount == 0 {
			// roll the dice
			input = config.PickInput()
		}

		// If we're debugging, log every 1/100th of the set as well as the final
		// state
		if config.Debug && (i%debugLogTick == 0 || i == iterations-1) {
			network.SetDebug(true)
		} else {
			network.SetDebug(false)
		}

		// input := config.Inputs[0]
		// oh shiiiiiiiiiiii
		// input := config.PickInput()
		network.Clear()
		network.Run(input.Values)
		var vals map[*Neuron]float64

		for j := len(network.GetLayers()) - 1; j >= 0; j-- {
			layer := network.GetLayers()[j]
			if layer == network.GetOutput() {
				vals = b.BackPropagateLayer(layer, b.BuildInputMap(input.Expected, network))
			} else {
				vals = b.BackPropagateLayer(layer, vals)
			}
		}

		if network.GetDebug() {
			totalError := 0.0
			for i, row := range input.Expected {
				for j, val := range row {
					totalError += math.Abs(
						Evaluator.LinearError(val, network.GetOutput().Neurons[i][j].Potential))
				}
			}

			Debug.Printf("Total error: %.3f\n", totalError)
		}
	}

}
