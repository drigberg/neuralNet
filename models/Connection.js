const ConnectionParams = require("./ConnectionParams")

class Connection {
    constructor({ in_neuron, out_neuron, weight }) {
        this.in_neuron = in_neuron
        this.out_neuron = out_neuron

        this.params = new ConnectionParams(weight)
    }
}

module.exports = Connection
