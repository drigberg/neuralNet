const Layer = require("./Layer")
const Neuron = require("./Neuron")
const Filter = require("./Filter")
const ConnectionParams = require("./ConnectionParams")
const Connection = require("./Connection")

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
        super({ in_layer, is_input, net, rectifier })

        this.neurons = {}
        this.filters = []
        this.architecture = []

        let neuron_args = { "layer": this }

        if (!is_input) {
            Object.assign(neuron_args, { in_neurons : in_layer.neurons })
        }

        for (var i = 0; i < depth; i++) {
            const filter = this.createFilterFromArchitecture({
                "net": this,
                "architecture": filter_structure
            })

            this.filters.push(filter)
        }

        this.generateArchitecture({
            "input_structure": in_layer.architecture,
            "filter_structure": filter_structure,
            "stride": stride
        })

        this.createFromArchitecture({
            "depth": depth,
            "layer": this,
            "filter_structure": filter_structure,
            "architecture": this.architecture,
            "neuron_args": neuron_args,
            "stride": stride
        })
    }

    generateArchitecture({ input_structure, filter_structure, stride }) {
        for (var j = 0; j < input_structure.length; j++) {
            let num_neurons = (input_structure[j] - (filter_structure[j] - stride)) / stride

            if (!Number.isInteger(num_neurons)) {
                let explanation = `(${input_structure[i]} - (${filter_structure[i]} - ${stride})) / ${stride} = ${num_neurons}`
                throw new Error(`Incompatible combination of input structure, filter structure, and stride: ${explanation}`)
            }
            this.architecture.push(num_neurons)
        }
    }

    createFromArchitecture({ architecture, filter_structure, layer, neuron_args, depth, stride }) {
        for (var filter_no = 0; filter_no < depth; filter_no++) {
            // prefix all neuron states with filter number (ex: "3x0.3.1")
            nest(architecture, null, null, filter_no)
        }

        function nest(arch, index, state, filter_no) {
            index = index || 0
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    let nested_state = String(state).includes("x") ? [state, i].join("."): [filter_no, i].join("x")
                    let nested = nest(arch, index + 1, nested_state, filter_no)
                }
            } else {
                let in_neurons = {}
                let args = Object.assign(neuron_args, { in_neurons })

                layer.neurons[state] = new Neuron(neuron_args)

                layer.createConnections({
                    "neuron": layer.neurons[state],
                    "filter_structure": filter_structure,
                    "filter_no": filter_no
                })

                return
            }
        }
    }

    createConnections({ neuron, filter_structure, filter_no }) {
        nest(filter_structure)

        function nest(arch, index, state) {
            index = index || 0
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    let nested_state = state ? [state, i].join("."): String(i)
                    let nested = nest(arch, index + 1, nested_state)
                }
            } else {
                neuron.connections.out[state] = new Connection(neuron.layer.filters[filter_no][state])
            }
        }
    }

    createFilterFromArchitecture({ architecture, net, filter_no }) {
        const obj = {}
        let sum = 0

        const layer = nest(architecture)

        let states = Object.keys(layer)

        const multiplier = 1 / sum

        for (var j = 0; j < states.length; j++) {
            obj[states[j]].weight *= multiplier
        }

        return obj

        function nest(arch, index, state) {
            index = index || 0
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    let nested_state = state ? [state, i].join("."): String(i)
                    let nested = nest(arch, index + 1, nested_state)
                }
            } else {
                const weight = Math.random()
                sum += weight
                obj[state] = new ConnectionParams(weight)
                return
            }
            return obj
        }
    }
}

module.exports = ConvolutionalLayer
