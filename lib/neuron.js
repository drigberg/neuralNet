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
     * @param {Neuron} options.inNeuron 
     * @param {Neuron} options.outNeuron 
     * @param {Number} options.weight 
     * @param {ConnectionParams} options.sharedParams 
     */
    constructor({ inNeuron, outNeuron, weight, sharedParams }) {
        this.inNeuron = inNeuron;
        this.outNeuron = outNeuron;

        if (weight) {
            this.params = new ConnectionParams(weight);
        } else if (sharedParams) {
            this.params = sharedParams;
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
                if (connection.inNeuron.activation > max.activation) {
                    max.activation = connection.inNeuron.activation;
                    max.derivative = connection.inNeuron.derivative;
                }
            });

            this.activation = max.activation;
            this.derivative = max.derivative;
        } else {
            let activation = this.bias;
            Object.values(this.connections.in).forEach((connection) => {
                activation += connection.inNeuron.activation * connection.params.weight;
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
                error += connection.outNeuron.error * connection.params.weight;
            });
            this.error = this.derivative * error;
        }

        // LEARN
        Object.values(this.connections.in).forEach((connection) => {
            const gradient = this.error * connection.inNeuron.activation;
            connection.params.weight += this.layer.net.learningRate * gradient;
        });

        this.bias += this.layer.net.learningRate * this.error;
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
