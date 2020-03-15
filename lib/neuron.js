/**
 * Module
 */

let neuronCount = 0;

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
 * Neuron
 * @class
 */
class Neuron {
    /**
     * Generates unique id for neuron
     * @static
     * @return {Number}
     */
    static uuid() {
        return neuronCount ++;
    }

    /**
     * 
     * @param {Object} options 
     * @param {Layer} options.layer
     * @param {Boolean} options.pooling
     */
    constructor({ layer, pooling = false }) {
        this._id = Neuron.uuid();
        this.layer = layer;
        this.pooling = pooling;
        this.bias = 0;
        this.connections = {
            out: {},
            in: {}
        };
    }

    /**
     * Determines new activation based on input connections
     */
    activate() {
        if (this.pooling) {
            const max = {
                derivative: null,
                activation: -Infinity
            };

            Object.values(this.connections.in).forEach((connection) => {
                if (connection.in_neuron.activation > max.activation) {
                    max.activation = connection.in_neuron.activation;
                    max.derivative = connection.in_neuron.derivative;
                }
            });

            this.activation = max.activation;
            this.derivative = max.derivative;
        } else {
            let activation = this.bias;
            Object.values(this.connections.in).forEach((connection) => {
                activation += connection.in_neuron.activation * connection.params.weight;
            });
        
            this.activation = this.layer.rectifier(activation, false);
            this.derivative = this.layer.rectifier(activation, true);
        }
    }

    /**
     * Adjusts bias based on error
     * @param {Array<Number>} target 
     */
    propagate(target) {
        if (typeof target === 'number') {
            this.error = target - this.activation;
        } else {
            let error = 0;

            Object.values(this.connections.out).forEach((connection) => {
                error += connection.out_neuron.error * connection.params.weight;
            });
            this.error = this.derivative * error;
        }

        // LEARN
        Object.values(this.connections.in).forEach((connection) => {
            const gradient = this.error * connection.in_neuron.activation;
            connection.params.weight += this.layer.net.learning_rate * gradient;
        });

        this.bias += this.layer.net.learning_rate * this.error;
    }
}

/**
 * Module exports
 */

module.exports = {
    Connection,
    ConnectionParams,
    Neuron
};
