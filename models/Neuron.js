/**
 * Module dependencies
 */

const {Connection} = require('./Connection');

let neurons = 0;

/**
 * Module
 */

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
        return neurons ++;
    }

    /**
     * 
     * @param {Object} options 
     * @param {Array<Neuron>} options.in_neurons
     * @param {Layer} options.layer
     * @param {Boolean} options.pooling
     */
    constructor({ in_neurons, layer, pooling }) {
        this._id = Neuron.uuid();
        this.layer = layer;
        this.pooling = pooling;
        this.bias = 0;

        this.connections = {
            out: {},
            in: {}
        };

        // create connections to all neurons in input layer
        if (in_neurons) {
            let sum = 0;
            this.bias = 0;

            Object.values(in_neurons).forEach((neuron) => {
                const weight = Math.random();
                sum += weight;

                const connection = new Connection({
                    in_neuron: neuron,
                    out_neuron: this,
                    weight
                });

                // update on both ends of the connection
                this.connections.in[neuron._id] = connection;
                neuron.connections.out[this._id] = connection;
            });

            // ensure that all weights add to 1
            const multiplier = 1 / sum;
            Object.values(this.connections.in).forEach((connection) => {
                connection.params.weight *= multiplier;
            });
        }
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
    Neuron
};
