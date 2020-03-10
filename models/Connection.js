/**
 * Module
 */

class ConnectionParams {
    constructor(weight) {
        this.weight = weight;
    }
}

class Connection {
    constructor({ in_neuron, out_neuron, weight, shared_params }) {
        this.in_neuron = in_neuron;
        this.out_neuron = out_neuron;

        if (weight) {
            this.params = new ConnectionParams(weight);
        } else if (shared_params) {
            this.params = shared_params;
        }
    }
}

/**
 * Module exports
 */

module.exports = {
    Connection,
    ConnectionParams
};
