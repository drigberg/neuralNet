const Layer = require("./Layer")
const Neuron = require("./Neuron")

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
        super({ in_layer, is_input, net, rectifier })

        let neuron_args = { "layer": this }

        if (!is_input) {
            Object.assign(neuron_args, { in_neurons : in_layer.neurons })
        }
        if (!architecture || !(architecture instanceof Array) || !architecture.length) {
            throw new Error("Architecture must be array with at least one dimension")
        }
        this.createFromArchitecture({
            "layer": this,
            "architecture": architecture,
            "neuron_args": neuron_args
        })
    }

    createFromArchitecture({ architecture, layer, neuron_args }) {
        layer.architecture = architecture
        layer.neurons = {}
        nest(architecture)

        function nest(arch, index, state) {
            index = index || 0
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    let nested_state = state ? [state, i].join("."): String(i)
                    let nested = nest(arch, index + 1, nested_state)
                }
            } else {
                layer.neurons[state] = new Neuron(neuron_args)
                return
            }
        }
    }
}

module.exports = FullyConnectedLayer
