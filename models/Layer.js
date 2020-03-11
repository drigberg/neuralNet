
/**
 * Module dependencies
 */

const {Neuron} = require('./Neuron');
const {Connection, ConnectionParams} = require('./Connection');
const errors = require('../lib/errors');

/**
 * Module
 */

/**
 * Layers contain neurons and expose functions for accessing them
 *
 * @class
 * @param {Number} net - the net to which the layer belongs
 * @param {Number} in_layer - layer forming incoming connections
 * @param {Boolean} is_input - true if layer is input layer
 * @param {Number} rectifier - activation function
 */
class Layer {
    constructor({ net, in_layer, is_input, rectifier }) {
        this.net = net;
        this.in_layer = in_layer;

        // rectifier is only used for non-input layers
        if (is_input) {
            this.rectifier = null;
        } else {
            this.rectifier = rectifier;
        }
    }

    activate() {
        for (var i = 0; i < this.neuronsAsArray.length; i++) {
            this.neuronsAsArray[i].activate();
        }

        return this.getActivations();
    }

    propagate(target_vector) {
        for (var i = 0; i < this.neuronsAsArray.length; i++) {
            let target = null;
            if (target_vector) {
                target = target_vector[i];
            }
            this.neuronsAsArray[i].propagate(target);
        }
    }

    getActivations() {
        return this.neuronsAsArray.map(neuron => neuron.activation);
    }

    get neuronsAsArray() {
        return Object.values(this.neurons);
    }
}

/**
 * Fully connected layers are expressed with one dimension of neurons
 *
 * @class
 * @param {Number} num_neurons - number of neurons in layer
 * @param {Number} in_layer - layer forming incoming connections
 * @param {Number} rectifier - activation function
 * @param {Number} net - the net to which the layer belongs
 * @param {Boolean} is_input - true if the layer represents the net's input
 */
class FullyConnectedLayer extends Layer {
    constructor({ architecture, in_layer, rectifier, net, is_input }) {
        super({ in_layer, is_input, net, rectifier });

        const neuron_args = { 'layer': this };

        if (!is_input) {
            neuron_args.in_neurons = in_layer.neurons;
        }

        if (!architecture || !(architecture instanceof Array) || !architecture.length) {
            throw new Error('Architecture must be array with at least one dimension');
        }

        const that = this;

        function assignNeuronsRecursive(arch, index, state) {
            index = index || 0;
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    const nested_state = state ? [state, i].join('.'): String(i);
                    assignNeuronsRecursive(arch, index + 1, nested_state);
                }
            } else {
                that.neurons[state] = new Neuron(neuron_args);
            }
        }

        this.architecture = architecture;
        this.neurons = {};
        assignNeuronsRecursive(architecture);
    }
}


/**
 *
 *
 * @class
 * @param {Number} num_neurons - number of neurons in layer
 * @param {Number} in_layer - layer forming incoming connections
 * @param {Number} rectifier - activation function
 * @param {Number} net - the net to which the layer belongs
 * @param {Boolean} is_input - true if the layer represents the net's input
 */
class ConvolutionalLayer extends Layer {
    constructor({ depth, filter_structure, stride, in_layer, rectifier, net, is_input }) {
        super({ in_layer, is_input, net, rectifier });

        this.neurons = {};
        this.filters = [];

        const neuron_args = { 'layer': this };

        if (!is_input) {
            neuron_args.in_neurons = in_layer.neurons;
        }

        for (var i = 0; i < depth; i++) {
            const filter = this.createFilterFromArchitecture({
                'net': this,
                'architecture': filter_structure
            });

            this.filters.push(filter);
        }

        this.architecture = this.generateArchitecture({
            'input_structure': in_layer.architecture,
            'filter_structure': filter_structure,
            'stride': stride
        });

        const that = this;

        function assignNeuronsRecursive(arch, index, state, filter_no) {
            index = index || 0;
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    let nested_state;
                    if (typeof state === 'string' && state.includes('x')) {
                        nested_state = [state, i].join('.');
                    } else {
                        nested_state = [filter_no, i].join('x');
                    }
                    assignNeuronsRecursive(arch, index + 1, nested_state, filter_no);
                }
            } else {
                neuron_args.in_neurons = {};

                that.neurons[state] = new Neuron(neuron_args);
                that.createConnections({
                    'neuron': that.neurons[state],
                    'filter_structure': filter_structure,
                    'filter_no': filter_no,
                    'in_layer': in_layer,
                    'neuron_state': state.split('x')[1],
                    'stride': stride,
                }); 
            }
        }

        for (var filter_no = 0; filter_no < depth; filter_no++) {
            // prefix all neuron states with filter number (ex: "3x0.3.1")
            assignNeuronsRecursive(this.architecture, null, null, filter_no);
        }
    }

    generateArchitecture({ input_structure, filter_structure, stride }) {
        const architecture = [];
        for (var j = 0; j < input_structure.length; j++) {
            const num_neurons = (input_structure[j] - (filter_structure[j] - stride)) / stride;

            if (!num_neurons || !Number.isInteger(num_neurons) || num_neurons < 1) {
                throw errors.errors.INCOMPATIBLE_FILTER({
                    'input_length': input_structure[j],
                    'filter_length': filter_structure[j],
                    'stride': stride,
                    'res': num_neurons
                });
            }
            architecture.push(num_neurons);
        }
        return architecture;
    }

    createConnection({ state, in_neuron_coords, in_layer, neuron, in_filter_no, filter_no }) {
        let in_neuron_key = in_neuron_coords.join('.');

        if (typeof in_filter_no === 'number') {
            in_neuron_key = [in_filter_no, 'x', in_neuron_key].join('');
        }

        const in_neuron = in_layer.neurons[in_neuron_key];

        const connection = new Connection({
            'in_neuron': in_neuron,
            'out_neuron': neuron,
            'shared_params': neuron.layer.filters[filter_no][state]
        });

        // update on both ends of the connection
        neuron.connections.in[in_neuron._id] = connection;
        in_neuron.connections.out[neuron._id] = connection;
    }

    createConnections({ neuron, filter_structure, filter_no, neuron_state, stride, in_layer }) {
        const num_in_filters = in_layer.filters ? in_layer.filters.length : 0;
        const that = this;

        function nest(arch, index, state) {
            index = index || 0;
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    const nested_state = state ? [state, i].join('.'): String(i);
                    nest(arch, index + 1, nested_state);
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

        nest(filter_structure);
    }

    createFilterFromArchitecture({ architecture }) {
        let sum = 0;
        const connectionParamsByState = {};

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
 * Module exports
 */

module.exports = {
    ConvolutionalLayer,
    FullyConnectedLayer,
};
