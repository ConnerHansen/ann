package main

import "time"

const (
	// TypeExcitatory the enum representing the excitatory neural type
	TypeExcitatory = iota

	// TypeInhibitory the enum representing the inhibitory neural type
	TypeInhibitory = iota
)

// Neuron the basic building block of the neural network structure
type Neuron struct {
	Bias      float64             `json:"bias"`
	FiredAt   time.Time           `json:"fired_at"`
	In        []*NeuronConnection `json:"incoming"`
	Out       []*NeuronConnection `json:"outgoing"`
	Potential float64             `json:"potential"`
	Type      int                 `json:"type"`
}

// NewNeuron returns a new base neuron with no connections
func NewNeuron(nType int) *Neuron {
	return &Neuron{
		Bias: 1.0,
		In:   make([]*NeuronConnection, 0),
		Out:  make([]*NeuronConnection, 0),
		Type: nType,
	}
}

// Clone clones the current neuron and its state, however it does not clone
// any of the connections to or from the source neuron
func (n *Neuron) Clone() *Neuron {
	clone := NewNeuron(n.Type)
	clone.FiredAt = n.FiredAt
	clone.Potential = n.Potential

	return clone
}

// Connect connects the given neuron to another neuron and then adds their
// connections to each neuron's appropriate connection list
func (n *Neuron) Connect(target *Neuron) *NeuronConnection {
	conn := NewNeuronConnection(n, target)

	n.AddOutgoing(conn)
	target.AddIncoming(conn)

	return conn
}

// AddIncoming adds a new incoming connection to the neuron. We allow multiple
// connections if desired, so uniqueness is not enforced
func (n *Neuron) AddIncoming(conn *NeuronConnection) {
	n.In = append(n.In, conn)
}

// AddOutgoing adds a new outgoing connection to the neuron. We allow multiple
// connections if desired, so uniqueness is not enforced
func (n *Neuron) AddOutgoing(conn *NeuronConnection) {
	n.Out = append(n.Out, conn)
}

// Fire fires the current neuron
func (n *Neuron) Fire() bool {
	// If there are no connections, we can't fire
	if len(n.Out) == 0 {
		return false
	}

	fired := false
	for _, conn := range n.Out {
		// If even one fires successfully, that would be the neuron firing
		fired = conn.Fire() || fired
	}

	if fired {
		n.FiredAt = time.Now()
		n.Potential = 0
	}

	return fired
}

// TotalInput calculates the total connection input on this neuron
func (n *Neuron) TotalInput() float64 {
	total := 0.0
	for _, conn := range n.In {
		total += conn.CalculateIntensity()
	}

	return total
}

// TotalInputWeight calculates the total connection input on this neuron
func (n *Neuron) TotalInputWeight() float64 {
	total := 0.0
	for _, conn := range n.In {
		total += conn.Weight
	}

	return total
}

// TotalOutput calculates the total connection output from this neuron
func (n *Neuron) TotalOutput() float64 {
	total := 0.0
	for _, conn := range n.Out {
		total += conn.CalculateIntensity()
	}

	return total
}

// TotalOutputWeight calculates the total connection output from this neuron
func (n *Neuron) TotalOutputWeight() float64 {
	total := 0.0
	for _, conn := range n.Out {
		total += conn.Weight
	}

	return total
}

// Output returns the calculated weight of the neuron which is the potential
// plus the bias
func (n *Neuron) Output() float64 {
	return n.Bias + n.Potential
}
