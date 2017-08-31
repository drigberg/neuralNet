class Connection {
    constructor(in_neuron, out_neuron, weight, bias) {
        Object.assign(this, {
            in_neuron,
            out_neuron,
            weight,
            bias
        })

        this.multiplyBy = (multiplier) => {
            this.weight *= multiplier
        }
    }
}

module.exports = Connection
