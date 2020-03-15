/**
 * Module
 */

/**
 * Parameters belonging to a connection
 */
class ConnectionParams {
    /**
     * 
     * @param {Number} weight 
     */
    constructor(weight) {
        this.weight = weight;
    }
}

/**
 * Connection between two neurons
 */
class Connection {
    /**
     * 
     * @param {Object} options
     * @param {Neuron} options.in_neuron 
     * @param {Neuron} options.out_neuron 
     * @param {Number} options.weight 
     * @param {ConnectionParams} options.shared_params 
     */
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
