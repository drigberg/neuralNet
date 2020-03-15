
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
     * @param {Boolean} is_input
     *  
     */
    constructor({ net, in_layer, rectifier }) {
        this.net = net;
        this.in_layer = in_layer;

        // rectifier is only used for non-input layers
        if (rectifier) {
            this.rectifier = rectifier;
        } else {
            this.rectifier = null;
        }
    }

    /**
     * Activates neurons and returns activations
     * @return {Array<Number>}
     */
    activate() {
        for (const [, neuron] of this.neuronGenerator()) {
            neuron.activate();
        }

        return this.activations;
    }

    /**
     * Adjusts weights based on error
     * @param {Array<Number>} target_vector 
     */
    propagate(target_vector) {
        for (const [i, neuron] of this.neuronGenerator()) {
            if (target_vector) {
                neuron.propagate(target_vector[i]);
            } else {
                neuron.propagate();
            }
        }
    }

    /**
     * @return {Array<Number>}
     */
    get activations() {
        return Object.values(this.neurons).map(neuron => neuron.activation);
    }

    /**
     * @return {Array<Neuron>}
     */
    *neuronGenerator() {
        const neurons = Object.values(this.neurons);
        for (let i = 0; i < neurons.length; i++) {
            yield [i, neurons[i]];
        }
        return neurons.length;
    }
}

/**
 * Fully connected layers are expressed with one dimension of neurons
 *
 * @class
 */
class FullyConnectedLayer extends Layer {
    /**
     * @param {Object} options
     * @param {Array<Number>} options.architecture - number of neurons in layers
     * @param {Layer} options.in_layer - layer forming incoming connections
     * @param {Function} options.rectifier - activation function
     * @param {Net} options.net - the net to which the layer belongs
     * @param {Boolean} options.is_input - true if the layer represents the net's input
     */
    constructor({ architecture, in_layer, rectifier, net, is_input }) {
        super({ in_layer, is_input, net, rectifier });

        this.architecture = architecture;

        const neuron_args = {
            layer: this
        };

        if (!is_input) {
            neuron_args.in_neurons = in_layer.neurons;
        }

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
                for (var i = 0; i < architecture[index]; i++) {
                    const nested_state = state ? [state, i].join('.'): String(i);
                    assignNeuronsRecursive(index + 1, nested_state);
                }
            } else {
                that.neurons[state] = new Neuron(neuron_args);
            }
        }

        this.neurons = {};
        assignNeuronsRecursive();
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
     * @param {Boolean} options.is_input - true if the layer represents the net's input
     * @param {Net} options.net - the net to which the layer belongs
     * @param {Function} options.rectifier - activation function
     * @param {Number} options.depth - depth
     * @param {Array<Number>} options.filter_architecture - architecture of filters
     * @param {Number} options.stride - distance between filters
     */
    constructor({ in_layer, is_input, net, rectifier, depth, filter_architecture, stride }) {
        super({ in_layer, is_input, net, rectifier });

        this.neurons = {};
        this.filters = [];

        const neuron_args = {
            layer: this
        };

        if (!is_input) {
            neuron_args.in_neurons = in_layer.neurons;
        }

        for (var i = 0; i < depth; i++) {
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
                for (var i = 0; i < that.architecture[index]; i++) {
                    let nested_state;
                    if (typeof state === 'string' && state.includes('x')) {
                        nested_state = [state, i].join('.');
                    } else {
                        nested_state = [filter_no, i].join('x');
                    }
                    assignNeuronsRecursive(index + 1, nested_state, filter_no);
                }
            } else {
                neuron_args.in_neurons = {};

                that.neurons[state] = new Neuron(neuron_args);
                that.createConnections({
                    neuron: that.neurons[state],
                    neuron_state: state.split('x')[1],
                    filter_architecture,
                    filter_no,
                    in_layer,
                    stride,
                }); 
            }
        }

        for (var filter_no = 0; filter_no < depth; filter_no++) {
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
        for (var j = 0; j < input_structure.length; j++) {
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

        const in_neuron = in_layer.neurons[in_neuron_key];

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
                for (var i = 0; i < filter_architecture[index]; i++) {
                    const nested_state = state ? [state, i].join('.'): String(i);
                    nest(index + 1, nested_state);
                }
            } else {
                const neuron_coords = neuron_state.split('.');
                const connection_coords = state.split('.');

                const in_neuron_coords = [];
                for (var j = 0; j < neuron_coords.length; j++) {
                    const coord = parseInt(neuron_coords[j], 10) * stride + parseInt(connection_coords[j], 10);
                    in_neuron_coords.push(coord);
                }

                if (num_in_filters > 0) {
                    for (var k = 0; k < num_in_filters; k++) {
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
                for (var i = 0; i < architecture[index]; i++) {
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
        const is_input = false;
        const rectifier = null;
        super({ in_layer, is_input, net, rectifier });

        const neuron_args = { 'layer': this, 'pooling': true };

        if (!spatial_extent || !(Number.isInteger(spatial_extent)) || spatial_extent < 2) {
            throw new Error('spatial_extent must be an integer greater than 1');
        }

        this.architecture = this.createPoolingArchitecture({ 
            input_architecture: in_layer.architecture,
            spatial_extent
        });

        this.createConnections({
            layer: this,
            neuron_args,
            in_layer,
            spatial_extent,
        });
    }

    /**
     * Creates connections
     * @param {Object} options 
     * @param {Layer} options.in_layer
     * @param {Layer} options.layer 
     * @param {Object} options.neuron_args 
     * @param {Number} options.spatial_extent 
     */
    createConnections({ in_layer, layer, neuron_args, spatial_extent }) {
        layer.neurons = {};
        const input_filters = in_layer.filters.length;

        /**
         * 
         * @param {Object} options 
         * @param {String} options.state
         * @param {Number} options.filter_no
         * @param {Number} options.spatial_extent
         */
        function getInputStates({ state, filter_no, spatial_extent }) {
            const base_state = state.split('.');
            base_state[0] *= spatial_extent;
            base_state[1] *= spatial_extent;

            const neuron_ids = [];
            let new_state;

            for (var i = 0; i < spatial_extent; i++) {
                // copy base_state
                new_state = base_state.slice();
                new_state[0] += i;
                for (var j = 0; j < spatial_extent; j++) {
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
            if (index < layer.architecture.length) {
                for (var i = 0; i < layer.architecture[index]; i++) {
                    const nested_state = state ? [state, i].join('.'): String(i);
                    nest(index + 1, nested_state);
                }
            } else {
                for (var filter_no = 0; filter_no < input_filters; filter_no++) {
                    const neuron_state = `${filter_no}x${state}`;

                    layer.neurons[neuron_state] = new Neuron(neuron_args);
                    const neuron = layer.neurons[neuron_state];

                    const input_states = getInputStates({ input_filters, state, filter_no, spatial_extent });

                    input_states.forEach((input_state) => {
                        const in_neuron = in_layer.neurons[input_state];

                        // connection must have weight 1 for error to propagate
                        const connection = new Connection({
                            out_neuron: neuron,
                            weight: 1,
                            in_neuron
                        });

                        // update on both ends of the connection
                        neuron.connections.in[neuron._id] = connection;
                        in_neuron.connections.out[in_neuron._id] = connection;
                    });
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

        for (var i = 0; i < input_architecture.length; i++) {
            if (i <= 1) {
                const dim = input_architecture[i] / spatial_extent;
                if (!Number.isInteger(dim)) {
                    const reason = `${input_architecture[i]} % ${spatial_extent} = ${dim}`;
                    throw new Error(`Pooling extent not compatible with input architecture! ${reason}`);
                }
                architecture.push(dim);
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
