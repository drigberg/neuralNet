class Connection {
    constructor(in_neuron, out_neuron, weight) {
        let bias = Math.random() * 0.1 - 0.2
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
