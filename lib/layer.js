
/**
 * Module dependencies
 */

const {Connection, ConnectionParams, Neuron} = require('./neuron');
const errors = require('./errors');

/**
 * Module
 */

/**
 * Layers contain neurons and expose functions for accessing them
 * @class
 */
class Layer {
    /**
     * 
     * @param {Object} options
     * @param {Net} net
     * @param {Layer} inLayer
     * @param {Function} rectifier
     *  
     */
    constructor({ net, inLayer, rectifier = null }) {
        this.net = net;
        this.inLayer = inLayer;
        this.rectifier = rectifier;
        this._states = null;
    }

    /**
     * Get all neuron states pre-fetched to save time on iteration
     */
    get states() {
        if (!this._states) {
            this._states = Object.keys(this.neuronsByState);
        }
        return this._states;
    }

    /**
     * Activates neurons and returns activations
     * @return {Array<Number>}
     */
    activate() {
        this.states.forEach((state) => {
            this.neuronsByState[state].activate();
        });

        return this.activations;
    }

    /**
     * Adjusts weights based on error
     * @param {Array<Number>} targetVector 
     */
    propagate(targetVector) {
        for (let i = 0; i < this.states.length; i++) {
            const state = this.states[i];
            if (targetVector) {
                this.neuronsByState[state].propagate(targetVector[i]);
            } else {
                this.neuronsByState[state].propagate();
            }
        }
    }

    /**
     * @return {Array<Number>}
     */
    get activations() {
        return this.states.map(state => this.neuronsByState[state].activation);
    }
}

/**
 * Fully connected layers have neurons which are connected to every neuron in the previous layer.
 * They are the only layer type which may be used as the input layer.
 *
 * @class
 */
class FullyConnectedLayer extends Layer {
    /**
     * @param {Object} options
     * @param {Array<Number>} options.architecture - number of neurons in layers
     * @param {Layer|null} options.inLayer - layer forming incoming connections
     * @param {Function} options.rectifier - activation function
     * @param {Net} options.net - the net to which the layer belongs
     */
    constructor({ architecture, inLayer, rectifier, net }) {
        super({ inLayer, net, rectifier });

        this.architecture = architecture;

        if (!architecture || !(architecture instanceof Array) || !architecture.length) {
            throw new Error('Architecture must be array with at least one dimension');
        }

        const that = this;

        /**
         * Recursive function for assigning neurons
         * @param {Number} index 
         * @param {String} state 
         */
        function assignNeuronsRecursive(index, state) {
            index = index || 0;
            if (index < architecture.length) {
                for (let i = 0; i < architecture[index]; i++) {
                    const nestedState = state ? [state, i].join('.'): String(i);
                    assignNeuronsRecursive(index + 1, nestedState);
                }
            } else {
                const neuron = new Neuron({ layer: that });
                if (inLayer) {
                    that.createConnections({ neuron, inLayer });
                }
                that.neuronsByState[state] = neuron;
            }
        }

        this.neuronsByState = {};
        assignNeuronsRecursive();
    }

    /**
     * Connects neuron to all neurons in previous layer
     * @param options - options object
     * @param options.neuron - new neuron
     * @param options.inLayer - previous layer
     */
    createConnections({ neuron, inLayer }) {
        // create connections to all neurons in input layer
        let sum = 0;

        inLayer.states.forEach((inNeuronState) => {
            const inNeuron = inLayer.neuronsByState[inNeuronState];
            const weight = Math.random();
            sum += weight;

            const connection = new Connection({
                inNeuron,
                outNeuron: neuron,
                weight
            });

            // update on both ends of the connection
            neuron.connections.in[inNeuron._id] = connection;
            inNeuron.connections.out[neuron._id] = connection;
        });

        // ensure that all weights add to 1
        const multiplier = 1 / sum;
        Object.values(neuron.connections.in).forEach((connection) => {
            connection.params.weight *= multiplier;
        });
    }
}


/**
 * Convolutional layer
 *
 * @class
 */
class ConvolutionalLayer extends Layer {
    /**
     * @param {Object} options
     * @param {Array<Number>} options.architecture - number of neurons in layers
     * @param {Layer} options.inLayer - layer forming incoming connections
     * @param {Net} options.net - the net to which the layer belongs
     * @param {Function} options.rectifier - activation function
     * @param {Number} options.depth - depth
     * @param {Array<Number>} options.filterArchitecture - architecture of filters
     * @param {Number} options.stride - distance between filters
     */
    constructor({ inLayer, net, rectifier, depth, filterArchitecture, stride }) {
        // only fully-connected layers can be input layer
        super({ inLayer, net, rectifier });

        this.neuronsByState = {};
        this.filters = [];

        for (let i = 0; i < depth; i++) {
            const filter = this.createFilterFromArchitecture({
                net: this,
                architecture: filterArchitecture
            });

            this.filters.push(filter);
        }

        this.architecture = this.generateArchitecture({
            inputStructure: inLayer.architecture,
            filterArchitecture,
            stride,
        });

        const that = this;

        /**
         * Recursive function for assigning neurons
         * @param {Number} index 
         * @param {String} state 
         * @param {Number} filterNumber 
         */
        function assignNeuronsRecursive(index, state, filterNumber) {
            index = index || 0;
            if (index < that.architecture.length) {
                for (let i = 0; i < that.architecture[index]; i++) {
                    let nestedState;
                    if (typeof state === 'string' && state.includes('x')) {
                        nestedState = [state, i].join('.');
                    } else {
                        nestedState = [filterNumber, i].join('x');
                    }
                    assignNeuronsRecursive(index + 1, nestedState, filterNumber);
                }
            } else {
                that.neuronsByState[state] = new Neuron({
                    layer: that
                });
                that.createConnections({
                    neuron: that.neuronsByState[state],
                    neuronState: state.split('x')[1],
                    filterArchitecture,
                    filterNumber,
                    inLayer,
                    stride,
                }); 
            }
        }

        for (let filterNumber = 0; filterNumber < depth; filterNumber++) {
            // prefix all neuron states with filter number (ex: "3x0.3.1")
            assignNeuronsRecursive(null, null, filterNumber);
        }
    }

    /**
     * Calculates architecture based on parameters
     * @param {Object} options 
     * @param {Array<Number>} inputStructure
     * @param {Array<Number>} filterArchitecture
     * @param {Number} stride
     */
    generateArchitecture({ inputStructure, filterArchitecture, stride }) {
        const architecture = [];
        for (let j = 0; j < inputStructure.length; j++) {
            const numNeurons = (inputStructure[j] - (filterArchitecture[j] - stride)) / stride;

            if (!numNeurons || !Number.isInteger(numNeurons) || numNeurons < 1) {
                throw errors.errors.INCOMPATIBLE_FILTER({
                    inputLength: inputStructure[j],
                    filterLength: filterArchitecture[j],
                    res: numNeurons,
                    stride,
                });
            }
            architecture.push(numNeurons);
        }
        return architecture;
    }

    /**
     * Creates connection between two neurons
     * @param {Object} options 
     * @param {String} options.state
     * @param {Array<Number>} options.inNeuronCoords
     * @param {Layer} options.inLayer
     * @param {Neuron} options.neuron
     * @param {Number} options.inFilterNumber
     * @param {Number} options.filterNumber

     */
    createConnection({ state, inNeuronCoords, inLayer, neuron, inFilterNumber, filterNumber }) {
        let inNeuronKey = inNeuronCoords.join('.');

        if (typeof inFilterNumber === 'number') {
            inNeuronKey = [inFilterNumber, 'x', inNeuronKey].join('');
        }

        const inNeuron = inLayer.neuronsByState[inNeuronKey];

        const connection = new Connection({
            inNeuron,
            outNeuron: neuron,
            sharedParams: neuron.layer.filters[filterNumber][state]
        });

        // update on both ends of the connection
        neuron.connections.in[inNeuron._id] = connection;
        inNeuron.connections.out[neuron._id] = connection;
    }

    /**
     * Creates connections to other layer
     * @param {Object} options
     * @param {Neuron} options.neuron
     * @param {Array<Number>} options.filterArchitecture
     * @param {Number} options.filterNumber
     * @param {String} options.neuronState
     * @param {Number} options.stride
     * @param {Layer} options.inLayer
     */
    createConnections({ neuron, filterArchitecture, filterNumber, neuronState, stride, inLayer }) {
        const numInFilters = inLayer.filters ? inLayer.filters.length : 0;
        const that = this;

        /**
         * 
         * @param {Number} index
         * @param {String} state 
         */
        function nest(index, state) {
            index = index || 0;
            if (index < filterArchitecture.length) {
                for (let i = 0; i < filterArchitecture[index]; i++) {
                    const nestedState = state ? [state, i].join('.'): String(i);
                    nest(index + 1, nestedState);
                }
            } else {
                const neuronCoords = neuronState.split('.');
                const connectionCoords = state.split('.');

                const inNeuronCoords = [];
                for (let j = 0; j < neuronCoords.length; j++) {
                    const coord = parseInt(neuronCoords[j], 10) * stride + parseInt(connectionCoords[j], 10);
                    inNeuronCoords.push(coord);
                }

                if (numInFilters > 0) {
                    for (let k = 0; k < numInFilters; k++) {
                        that.createConnection({ state, inNeuronCoords, inLayer, neuron, filterNumber, inFilterNumber: k });
                    }
                } else {
                    that.createConnection({ state, inNeuronCoords, inLayer, neuron, filterNumber, inFilterNumber: null });
                }
            }
        }

        nest();
    }

    /**
     * Creates filter based on structure
     * @param {Object} options 
     * @param {Array<Number>} options.architecture
     */
    createFilterFromArchitecture({ architecture }) {
        let sum = 0;
        const connectionParamsByState = {};

        /**
         * 
         * @param {Number} index
         * @param {String} state 
         */
        function nest(index, state) {
            index = index || 0;
            if (index < architecture.length) {
                for (let i = 0; i < architecture[index]; i++) {
                    const nestedState = state ? [state, i].join('.'): String(i);
                    nest(index + 1, nestedState);
                }
            } else {
                const weight = Math.random();
                sum += weight;
                connectionParamsByState[state] = new ConnectionParams(weight);
            }
        }

        nest();

        const multiplier = 1 / sum;
        Object.values(connectionParamsByState).forEach((connectionParams) => {
            connectionParams.weight *= multiplier;
        });

        return connectionParamsByState;
    }
}

/**
 * Fully connected layers are expressed with one dimension of neurons
 *
 * @class
 */
class PoolingLayer extends Layer {
    /**
     * 
     * @param {Object} options 
     * @param {Number} options.spatialExtent
     * @param {Layer} options.inLayer
     * @param {Net} options.net
     */
    constructor({ spatialExtent, inLayer, net }) {
        const rectifier = null;
        super({ inLayer, net, rectifier });

        if (!spatialExtent || !(Number.isInteger(spatialExtent)) || spatialExtent < 2) {
            throw new Error('spatialExtent must be an integer greater than 1');
        }

        this.architecture = this.createPoolingArchitecture({ 
            inputArchitecture: inLayer.architecture,
            spatialExtent
        });

        this.createConnections({
            inLayer,
            spatialExtent,
        });
    }

    /**
     * Creates connections
     * @param {Object} options 
     * @param {Layer} options.inLayer
     * @param {Number} options.spatialExtent 
     */
    createConnections({ inLayer, spatialExtent }) {
        const that = this;
        this.neuronsByState = {};
        const inputFilters = inLayer.filters.length;

        /**
         * 
         * @param {Object} options 
         * @param {String} options.state
         * @param {Number} options.filterNumber
         */
        function getInputStates({ state, filterNumber }) {
            const baseState = state.split('.');
            baseState[0] *= spatialExtent;
            baseState[1] *= spatialExtent;

            const neuronStates = [];
            let newState;

            for (let i = 0; i < spatialExtent; i++) {
                // copy baseState
                newState = baseState.slice();
                newState[0] += i;
                for (let j = 0; j < spatialExtent; j++) {
                    newState[1] += j;
                    const neuronState = `${filterNumber}x${newState.join('.')}`;
                    neuronStates.push(neuronState);
                }
            }

            return neuronStates;
        }

        /**
         * 
         * @param {Number} index 
         * @param {String} state
         */
        function nest(index, state) {
            index = index || 0;
            if (index < that.architecture.length) {
                for (let i = 0; i < that.architecture[index]; i++) {
                    const nestedState = state ? [state, i].join('.'): String(i);
                    nest(index + 1, nestedState);
                }
            } else {
                for (let filterNumber = 0; filterNumber < inputFilters; filterNumber++) {
                    const neuron = new Neuron({
                        layer: that,
                        pooling: true
                    });

                    const inputStates = getInputStates({ inputFilters, state, filterNumber });

                    inputStates.forEach((inputState) => {
                        const inNeuron = inLayer.neuronsByState[inputState];

                        // connection must have weight 1 for error to propagate
                        const connection = new Connection({
                            outNeuron: neuron,
                            weight: 1,
                            inNeuron
                        });

                        // update on both ends of the connection
                        neuron.connections.in[inNeuron._id] = connection;
                        inNeuron.connections.out[neuron._id] = connection;
                    });

                    const neuronState = `${filterNumber}x${state}`;
                    that.neuronsByState[neuronState] = neuron;
                }
            }
        }

        nest();
    }

    /**
     * 
     * @param {Object} options
     * @param {Array<Number>} options.inputArchitecture
     * @param {Number} options.spatialExtent
     */
    createPoolingArchitecture({ inputArchitecture, spatialExtent }) {
        const architecture = [];

        for (let i = 0; i < inputArchitecture.length; i++) {
            if (i <= 1) {
                const dimension = inputArchitecture[i] / spatialExtent;
                if (!Number.isInteger(dimension)) {
                    const reason = `${inputArchitecture[i]} % ${spatialExtent} = ${dimension}`;
                    throw new Error(`Pooling extent not compatible with input architecture! ${reason}`);
                }
                architecture.push(dimension);
            } else {
                architecture.push(inputArchitecture[i]);
            }
        }
        return architecture;
    }
}


/**
 * Module exports
 */

module.exports = {
    ConvolutionalLayer,
    FullyConnectedLayer,
    PoolingLayer,
};
