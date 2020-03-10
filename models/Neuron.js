/**
 * Module dependencies
 */

const {Connection} = require('./Connection');

let neurons = 0;

/**
 * Module
 */

class Neuron {
    static uuid() {
        return neurons ++;
    }

    constructor({ in_neurons, layer }) {
        this._id = Neuron.uuid();
        this.layer = layer;

        this.connections = {
            'out': {},
            'in': {}
        };

        // create connections to all neurons in input layer
        if (in_neurons) {
            let sum = 0;
            this.bias = 0;

            const neuron_keys = Object.keys(in_neurons);
            neuron_keys.forEach((key) => {
                const neuron = in_neurons[key];
                const weight = Math.random();
                sum += weight;

                const connection = new Connection({
                    'in_neuron': neuron,
                    'out_neuron': this,
                    'weight': weight
                });

                // update on both ends of the connection
                this.connections.in[neuron._id] = connection;
                neuron.connections.out[this._id] = connection;
            });

            // ensure that all weights add to 1
            const multiplier = 1 / sum;
            const keys = Object.keys(this.connections.in);
            for (var j = 0; j < keys.length; j++) {
                const key = keys[j];
                this.connections.in[key].params.weight *= multiplier;
            }
        }
    }

    activate() {
        let activation = this.bias;
        const connections = this.connections.in;
        const keys = Object.keys(connections);

        for (var i = 0; i < keys.length; i++) {
            const connection = connections[keys[i]];
            activation += connection.in_neuron.activation * connection.params.weight;
        }

        this.activation = this.layer.rectifier(activation, false);
        this.derivative = this.layer.rectifier(activation, true);
    }

    propagate(target) {
        if (typeof target === 'number') {
            this.error = target - this.activation;
        } else {
            let error = 0;

            const connections = this.connections.out;
            const keys = Object.keys(connections);

            keys.forEach((key) => {
                const connection = connections[key];
                error += connection.out_neuron.error * connection.params.weight;
            });
            this.error = this.derivative * error;
        }

        // LEARN
        const keys = Object.keys(this.connections.in);
        for (var j = 0; j < keys.length; j++) {
            const connection = this.connections.in[keys[j]];
            const gradient = this.error * connection.in_neuron.activation;
            connection.params.weight += this.layer.net.learning_rate * gradient;
        }

        this.bias += this.layer.net.learning_rate * this.error;
    }
}

/**
 * Module exports
 */

module.exports = Neuron;
