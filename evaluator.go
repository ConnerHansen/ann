package main

import "math"

// Evaluator is the default evaluator to be used across the neural network
var Evaluator = &DefaultEvaluator{}

// DefaultEvaluator is the default evaluator for our neural networks
type DefaultEvaluator struct{}

// AdjustLayer performs the actual fine tuning of the current layer given a base
// error mapping. This returns the error mapping for the current layer
func (e *DefaultEvaluator) AdjustLayer(layer *NetworkLayer, errMap map[*Neuron]*NeuronError) (map[*Neuron]*NeuronError, error) {
	currErrMap := make(map[*Neuron]*NeuronError)

	// Go through each neuron in the current layer and adjust the outgoing
	// weights accordingly. Track how much adjustment it takes overall
	layer.EachNeuron(func(n *Neuron) {
		currErrMap[n] = &NeuronError{Direction: 1}

		// Get the total weight coming in to this neuron and stored that for future
		// usage
		for _, conn := range n.In {
			currErrMap[n].TotalWeight += conn.Weight
		}

		// Now, figure out how much to adjust the incoming weights
		for _, conn := range n.Out {
			err := errMap[conn.Target]

			// How much of the error is this connection responsible for?
			proportionalWeight := conn.Weight / conn.Target.TotalInputWeight()

			// Now, it's proportional, but we need to adjust from gross to fine tuning
			// and taking our current weight into account alongside the sigmoid helps.
			// Right now we've got the scaling value cranked all the way up because it
			// makes things converge more rapidly
			adjStep := 0.1 * proportionalWeight * err.Sigmoid()
			// adjStep := 0.2 * conn.Weight * proportionalWeight * err.Sigmoid()

			// Keep track of how much error we have at this layer so we can percolate
			// that up
			currErrMap[n].Error += adjStep

			// Now do the actual adjustment
			conn.Weight += adjStep * float64(err.Direction)
		}

		// Make sure we keep our logic consistent (ie separate magnitude and direction)
		if currErrMap[n].Error < 0 {
			currErrMap[n].Direction = -1
			currErrMap[n].Error *= -1
		}

		// Make sure we scale the error appropriately
		// currErrMap[n].Error = e.MeanSquaredError(0.0, currErrMap[n].Error)
	})

	return currErrMap, nil
}

// PerformBackPropagation performs traditional back propagation of the network signal
func (e *DefaultEvaluator) PerformBackPropagation(expected [][]float64, network NetworkConfiguration) error {
	baseError, err := e.CalculateError(expected, network)

	if err != nil {
		Error.Println("Error while attempting to backpropagate:", err)
		return err
	}

	layers := network.GetLayers()

	// No point trying to adjust the outgoing weights on the last layer amirite
	for i := len(layers) - 2; i >= 0; i-- {
		layer := layers[i]
		baseError, err = e.AdjustLayer(layer, baseError)

		// Make sure we capture any failures in the layers
		if err != nil {
			Error.Println("Error while attempting to backpropagate:", err)
			return err
		}
	}

	return nil
}

// CalculateError calculates the error of the output layer versuses the provided
// set of expected values
func (e *DefaultEvaluator) CalculateError(expected [][]float64, network NetworkConfiguration) (map[*Neuron]*NeuronError, error) {
	layer := network.GetOutput()

	if len(expected) != len(layer.Neurons) ||
		len(expected[0]) != len(layer.Neurons[0]) {
		return nil, ErrArraySizeMismatch
	}

	errMap := make(map[*Neuron]*NeuronError)

	for i, row := range expected {
		for j, val := range row {
			weight := 0.0
			neuron := layer.Neurons[i][j]
			for _, input := range neuron.In {
				weight += input.Weight
			}

			direction := 1
			if val < neuron.Potential {
				direction = -1
			}

			err := &NeuronError{
				Direction:   direction,
				Error:       e.MeanSquaredError(val, neuron.Potential),
				TotalWeight: weight,
			}
			errMap[neuron] = err
		}
	}

	return errMap, nil
}

// Train runs the given network with
func (e *DefaultEvaluator) Train(iterations int, config *TrainingConfiguration) {
	// func (e *DefaultEvaluator) Train(iterations int, input, expected [][]float64, network NetworkConfiguration) {
	network := config.Network

	// Log 100 frames if we're debugging
	debugLogTick := iterations / 100
	for i := 0; i < iterations; i++ {
		// If we're debugging, log every 1/100th of the set as well as the final
		// state
		if config.Debug && (i%debugLogTick == 0 || i == iterations-1) {
			network.SetDebug(true)
		} else {
			network.SetDebug(false)
		}
		input := config.PickInput()
		network.Run(input.Values)
		e.PerformBackPropagation(input.Expected, network)

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

// LinearError calculates the linear error between two float values
func (e *DefaultEvaluator) LinearError(expected, actual float64) float64 {
	return actual - expected
}

// MeanSquaredError calculates the mean squared error between two float values
func (e *DefaultEvaluator) MeanSquaredError(expected, actual float64) float64 {
	return 0.5 * math.Pow(expected-actual, 2)
}
