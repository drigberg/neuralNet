const Layer = require('./Layer');
const Neuron = require('./Neuron');
const ConnectionParams = require('./ConnectionParams');
const Connection = require('./Connection');
const errors = require('../lib/errors');

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
        this.architecture = [];

        const neuron_args = { 'layer': this };

        if (!is_input) {
            Object.assign(neuron_args, { in_neurons : in_layer.neurons });
        }

        for (var i = 0; i < depth; i++) {
            const filter = this.createFilterFromArchitecture({
                'net': this,
                'architecture': filter_structure
            });

            this.filters.push(filter);
        }

        this.generateArchitecture({
            'input_structure': in_layer.architecture,
            'filter_structure': filter_structure,
            'stride': stride
        });

        this.createFromArchitecture({
            'depth': depth,
            'layer': this,
            'in_layer': in_layer,
            'filter_structure': filter_structure,
            'architecture': this.architecture,
            'neuron_args': neuron_args,
            'stride': stride
        });
    }

    generateArchitecture({ input_structure, filter_structure, stride }) {
        for (var j = 0; j < input_structure.length; j++) {
            const num_neurons = (input_structure[j] - (filter_structure[j] - stride)) / stride;

            if (!num_neurons || !Number.isInteger(num_neurons) || num_neurons < 1) {
                throw errors.INCOMPATIBLE_FILTER({
                    'input_length': input_structure[j],
                    'filter_length': filter_structure[j],
                    'stride': stride,
                    'res': num_neurons
                });
            }
            this.architecture.push(num_neurons);
        }
    }

    createFromArchitecture({ architecture, filter_structure, layer, neuron_args, depth, stride, in_layer }) {
        for (var filter_no = 0; filter_no < depth; filter_no++) {
            // prefix all neuron states with filter number (ex: "3x0.3.1")
            nest(architecture, null, null, filter_no);
        }

        function nest(arch, index, state, filter_no) {
            index = index || 0;
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    const nested_state = String(state).includes('x') ? [state, i].join('.'): [filter_no, i].join('x');
                    nest(arch, index + 1, nested_state, filter_no);
                }
            } else {
                const in_neurons = {};
                Object.assign(neuron_args, { in_neurons });

                layer.neurons[state] = new Neuron(neuron_args);

                layer.createConnections({
                    'neuron': layer.neurons[state],
                    'filter_structure': filter_structure,
                    'filter_no': filter_no,
                    'in_layer': in_layer,
                    'neuron_state': state.split('x')[1],
                    'stride': stride,
                });

                
            }
        }
    }

    createConnections({ neuron, filter_structure, filter_no, neuron_state, stride, in_layer }) {
        const in_filters = in_layer.filters ? in_layer.filters.length : 0;

        nest(filter_structure);

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
                    const coord = Number(neuron_coords[j]) * stride + Number(connection_coords[j]);
                    in_neuron_coords.push(coord);
                }

                if (in_filters) {
                    for (var k = 0; k < in_filters.length; k++) {
                        const in_filter_no = k;
                        createConnection({ state, in_neuron_coords, in_layer, neuron, in_filter_no });
                    }
                } else {
                    createConnection({ state, in_neuron_coords, in_layer, neuron });
                }
            }
        }


        function createConnection({ state, in_neuron_coords, in_layer, neuron, in_filter_no }) {
            let in_neuron_key = in_neuron_coords.join('.');

            in_neuron_key = in_filter_no ? [in_filter_no, 'x', in_neuron_key].join('') : in_neuron_key;

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
    }

    createFilterFromArchitecture({ architecture }) {
        const obj = {};
        let sum = 0;

        const layer = nest(architecture);

        const states = Object.keys(layer);

        const multiplier = 1 / sum;

        for (var j = 0; j < states.length; j++) {
            obj[states[j]].weight *= multiplier;
        }

        return obj;

        function nest(arch, index, state) {
            index = index || 0;
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    const nested_state = state ? [state, i].join('.'): String(i);
                    nest(arch, index + 1, nested_state);
                }
            } else {
                const weight = Math.random();
                sum += weight;
                obj[state] = new ConnectionParams(weight);
                return;
            }
            return obj;
        }
    }
}

module.exports = ConvolutionalLayer;
