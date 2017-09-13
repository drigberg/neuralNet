const Layer = require("./Layer")
const Neuron = require("./Neuron")
const Connection = require("./Connection")


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
class PoolingLayer extends Layer {
    constructor({ spatial_extent, in_layer, net }) {
        let is_input = false
        let rectifier = null
        super({ in_layer, is_input, net, rectifier })

        let neuron_args = { "layer": this, "pooling": true }

        if (!spatial_extent || !(Number.isInteger(spatial_extent)) || spatial_extent < 2) {
            throw new Error("spatial_extent must be an integer greater than 1")
        }

        this.getPoolingArchitecture({ "input_architecture": in_layer.architecture, "spatial_extent": spatial_extent })

        this.createFromArchitecture({
            "layer": this,
            "architecture": this.architecture,
            "in_layer": in_layer,
            "input_architecture": in_layer.architecture,
            "neuron_args": neuron_args,
            "spatial_extent": spatial_extent
        })
    }

    createFromArchitecture({ architecture, in_layer, layer, neuron_args, input_architecture, spatial_extent }) {
        layer.architecture = architecture
        layer.neurons = {}
        let input_filters = in_layer.filters.length

        nest(architecture)

        function nest(arch, index, state) {
            index = index || 0
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    let nested_state = state ? [state, i].join("."): String(i)
                    let nested = nest(arch, index + 1, nested_state)
                }
            } else {
                for (var filter_no = 0; filter_no < input_filters; filter_no++) {
                    let neuron_state = `${filter_no}x${state}`

                    layer.neurons[neuron_state] = new Neuron(neuron_args)
                    let neuron = layer.neurons[neuron_state]

                    let input_states = getInputStates({ input_filters, state, filter_no, spatial_extent })

                    input_states.forEach((input_state) => {
                        let in_neuron = in_layer.neurons[input_state]

                        // connection must have weight 1 for error to propagate
                        let connection = new Connection({
                            "in_neuron": in_neuron,
                            "out_neuron": neuron,
                            "weight": 1
                        })

                        // update on both ends of the connection
                        neuron.connections.in[neuron._id] = connection
                        in_neuron.connections.out[in_neuron._id] = connection
                    })
                }
            }
        }

        function getInputStates({ input_filters, state, filter_no, spatial_extent }) {
            let base_state = state.split(".")
            base_state[0] *= spatial_extent
            base_state[1] *= spatial_extent

            let neuron_ids = []
            let new_state

            for (var i = 0; i < spatial_extent; i++) {
                // copy base_state
                new_state = base_state.slice()
                new_state[0] += i
                for (var j = 0; j < spatial_extent; j++) {
                    new_state[1] += j
                    let neuron_id = `${filter_no}x${new_state.join(".")}`
                    neuron_ids.push(neuron_id)
                }
            }

            return neuron_ids
        }
    }

    getPoolingArchitecture({ input_architecture, spatial_extent }) {
        this.architecture = []

        for (var i = 0; i < input_architecture.length; i++) {
            if (i <= 1) {
                let dim = input_architecture[i] / spatial_extent
                if (!Number.isInteger(dim)) {
                    let reason = `${input_architecture[i]} % ${spatial_extent} = ${dim}`
                    throw new Error(`Pooling extent not compatible with input architecture! ${reason}`)
                }
                this.architecture.push(dim)
            } else {
                this.architecture.push(input_architecture[i])
            }
        }
    }
}

module.exports = PoolingLayer
