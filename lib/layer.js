
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
     * @param {Layer} in_layer
     * @param {Function} rectifier
     *  
     */
    constructor({ net, in_layer, rectifier = null }) {
        this.net = net;
        this.in_layer = in_layer;
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
     * @param {Array<Number>} target_vector 
     */
    propagate(target_vector) {
        for (let i = 0; i < this.states.length; i++) {
            const state = this.states[i];
            if (target_vector) {
                this.neuronsByState[state].propagate(target_vector[i]);
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
     * @param {Layer|null} options.in_layer - layer forming incoming connections
     * @param {Function} options.rectifier - activation function
     * @param {Net} options.net - the net to which the layer belongs
     */
    constructor({ architecture, in_layer, rectifier, net }) {
        super({ in_layer, net, rectifier });

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
                    const nested_state = state ? [state, i].join('.'): String(i);
                    assignNeuronsRecursive(index + 1, nested_state);
                }
            } else {
                const neuron = new Neuron({ layer: that });
                if (in_layer) {
                    that.createConnections({ neuron, in_layer });
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
     * @param options.in_layer - previous layer
     */
    createConnections({ neuron, in_layer }) {
        // create connections to all neurons in input layer
        let sum = 0;

        in_layer.states.forEach((in_neuron_state) => {
            const in_neuron = in_layer.neuronsByState[in_neuron_state];
            const weight = Math.random();
            sum += weight;

            const connection = new Connection({
                in_neuron: in_neuron,
                out_neuron: neuron,
                weight
            });

            // update on both ends of the connection
            neuron.connections.in[in_neuron._id] = connection;
            in_neuron.connections.out[neuron._id] = connection;
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
     * @param {Layer} options.in_layer - layer forming incoming connections
     * @param {Net} options.net - the net to which the layer belongs
     * @param {Function} options.rectifier - activation function
     * @param {Number} options.depth - depth
     * @param {Array<Number>} options.filter_architecture - architecture of filters
     * @param {Number} options.stride - distance between filters
     */
    constructor({ in_layer, net, rectifier, depth, filter_architecture, stride }) {
        // only fully-connected layers can be input layer
        super({ in_layer, net, rectifier });

        this.neuronsByState = {};
        this.filters = [];

        for (let i = 0; i < depth; i++) {
            const filter = this.createFilterFromArchitecture({
                net: this,
                architecture: filter_architecture
            });

            this.filters.push(filter);
        }

        this.architecture = this.generateArchitecture({
            input_structure: in_layer.architecture,
            filter_architecture,
            stride,
        });

        const that = this;

        /**
         * Recursive function for assigning neurons
         * @param {Number} index 
         * @param {String} state 
         * @param {Number} filter_no 
         */
        function assignNeuronsRecursive(index, state, filter_no) {
            index = index || 0;
            if (index < that.architecture.length) {
                for (let i = 0; i < that.architecture[index]; i++) {
                    let nested_state;
                    if (typeof state === 'string' && state.includes('x')) {
                        nested_state = [state, i].join('.');
                    } else {
                        nested_state = [filter_no, i].join('x');
                    }
                    assignNeuronsRecursive(index + 1, nested_state, filter_no);
                }
            } else {
                that.neuronsByState[state] = new Neuron({
                    layer: that
                });
                that.createConnections({
                    neuron: that.neuronsByState[state],
                    neuron_state: state.split('x')[1],
                    filter_architecture,
                    filter_no,
                    in_layer,
                    stride,
                }); 
            }
        }

        for (let filter_no = 0; filter_no < depth; filter_no++) {
            // prefix all neuron states with filter number (ex: "3x0.3.1")
            assignNeuronsRecursive(null, null, filter_no);
        }
    }

    /**
     * Calculates architecture based on parameters
     * @param {Object} options 
     * @param {Array<Number>} input_structure
     * @param {Array<Number>} filter_architecture
     * @param {Number} stride
     */
    generateArchitecture({ input_structure, filter_architecture, stride }) {
        const architecture = [];
        for (let j = 0; j < input_structure.length; j++) {
            const num_neurons = (input_structure[j] - (filter_architecture[j] - stride)) / stride;

            if (!num_neurons || !Number.isInteger(num_neurons) || num_neurons < 1) {
                throw errors.errors.INCOMPATIBLE_FILTER({
                    input_length: input_structure[j],
                    filter_length: filter_architecture[j],
                    res: num_neurons,
                    stride,
                });
            }
            architecture.push(num_neurons);
        }
        return architecture;
    }

    /**
     * Creates connection between two neurons
     * @param {Object} options 
     * @param {String} options.state
     * @param {Array<Number>} options.in_neuron_coords
     * @param {Layer} options.in_layer
     * @param {Neuron} options.neuron
     * @param {Number} options.in_filter_no
     * @param {Number} options.filter_no

     */
    createConnection({ state, in_neuron_coords, in_layer, neuron, in_filter_no, filter_no }) {
        let in_neuron_key = in_neuron_coords.join('.');

        if (typeof in_filter_no === 'number') {
            in_neuron_key = [in_filter_no, 'x', in_neuron_key].join('');
        }

        const in_neuron = in_layer.neuronsByState[in_neuron_key];

        const connection = new Connection({
            in_neuron,
            out_neuron: neuron,
            shared_params: neuron.layer.filters[filter_no][state]
        });

        // update on both ends of the connection
        neuron.connections.in[in_neuron._id] = connection;
        in_neuron.connections.out[neuron._id] = connection;
    }

    /**
     * Creates connections to other layer
     * @param {Object} options
     * @param {Neuron} options.neuron
     * @param {Array<Number>} options.filter_architecture
     * @param {Number} options.filter_no
     * @param {String} options.neuron_state
     * @param {Number} options.stride
     * @param {Layer} options.in_layer
     */
    createConnections({ neuron, filter_architecture, filter_no, neuron_state, stride, in_layer }) {
        const num_in_filters = in_layer.filters ? in_layer.filters.length : 0;
        const that = this;

        /**
         * 
         * @param {Number} index
         * @param {String} state 
         */
        function nest(index, state) {
            index = index || 0;
            if (index < filter_architecture.length) {
                for (let i = 0; i < filter_architecture[index]; i++) {
                    const nested_state = state ? [state, i].join('.'): String(i);
                    nest(index + 1, nested_state);
                }
            } else {
                const neuron_coords = neuron_state.split('.');
                const connection_coords = state.split('.');

                const in_neuron_coords = [];
                for (let j = 0; j < neuron_coords.length; j++) {
                    const coord = parseInt(neuron_coords[j], 10) * stride + parseInt(connection_coords[j], 10);
                    in_neuron_coords.push(coord);
                }

                if (num_in_filters > 0) {
                    for (let k = 0; k < num_in_filters; k++) {
                        that.createConnection({ state, in_neuron_coords, in_layer, neuron, filter_no, in_filter_no: k });
                    }
                } else {
                    that.createConnection({ state, in_neuron_coords, in_layer, neuron, filter_no, in_filter_no: null });
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
                    const nested_state = state ? [state, i].join('.'): String(i);
                    nest(index + 1, nested_state);
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
     * @param {Number} options.spatial_extent
     * @param {Layer} options.in_layer
     * @param {Net} options.net
     */
    constructor({ spatial_extent, in_layer, net }) {
        const rectifier = null;
        super({ in_layer, net, rectifier });

        if (!spatial_extent || !(Number.isInteger(spatial_extent)) || spatial_extent < 2) {
            throw new Error('spatial_extent must be an integer greater than 1');
        }

        this.architecture = this.createPoolingArchitecture({ 
            input_architecture: in_layer.architecture,
            spatial_extent
        });

        this.createConnections({
            in_layer,
            spatial_extent,
        });
    }

    /**
     * Creates connections
     * @param {Object} options 
     * @param {Layer} options.in_layer
     * @param {Number} options.spatial_extent 
     */
    createConnections({ in_layer, spatial_extent }) {
        const that = this;
        this.neuronsByState = {};
        const input_filters = in_layer.filters.length;

        /**
         * 
         * @param {Object} options 
         * @param {String} options.state
         * @param {Number} options.filter_no
         */
        function getInputStates({ state, filter_no }) {
            const base_state = state.split('.');
            base_state[0] *= spatial_extent;
            base_state[1] *= spatial_extent;

            const neuron_ids = [];
            let new_state;

            for (let i = 0; i < spatial_extent; i++) {
                // copy base_state
                new_state = base_state.slice();
                new_state[0] += i;
                for (let j = 0; j < spatial_extent; j++) {
                    new_state[1] += j;
                    const neuron_id = `${filter_no}x${new_state.join('.')}`;
                    neuron_ids.push(neuron_id);
                }
            }

            return neuron_ids;
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
                    const nested_state = state ? [state, i].join('.'): String(i);
                    nest(index + 1, nested_state);
                }
            } else {
                for (let filter_no = 0; filter_no < input_filters; filter_no++) {
                    const neuron = new Neuron({
                        layer: that,
                        pooling: true
                    });

                    const input_states = getInputStates({ input_filters, state, filter_no });

                    input_states.forEach((input_state) => {
                        const in_neuron = in_layer.neuronsByState[input_state];

                        // connection must have weight 1 for error to propagate
                        const connection = new Connection({
                            out_neuron: neuron,
                            weight: 1,
                            in_neuron
                        });

                        // update on both ends of the connection
                        neuron.connections.in[in_neuron._id] = connection;
                        in_neuron.connections.out[neuron._id] = connection;
                    });

                    const neuron_state = `${filter_no}x${state}`;
                    that.neuronsByState[neuron_state] = neuron;
                }
            }
        }

        nest();
    }

    /**
     * 
     * @param {Object} options
     * @param {Array<Number>} options.input_architecture
     * @param {Number} options.spatial_extent
     */
    createPoolingArchitecture({ input_architecture, spatial_extent }) {
        const architecture = [];

        for (let i = 0; i < input_architecture.length; i++) {
            if (i <= 1) {
                const dimension = input_architecture[i] / spatial_extent;
                if (!Number.isInteger(dimension)) {
                    const reason = `${input_architecture[i]} % ${spatial_extent} = ${dimension}`;
                    throw new Error(`Pooling extent not compatible with input architecture! ${reason}`);
                }
                architecture.push(dimension);
            } else {
                architecture.push(input_architecture[i]);
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
