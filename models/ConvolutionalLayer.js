const Layer = require("./Layer")
const Neuron = require("./Neuron")
const Filter = require("./Filter")

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
        // super({ in_layer, is_input, net, rectifier })
        //
        // this.filters = []
        // for (var i = 0; i < depth; i++) {
        //     this.filters.push(new Filter(filter_structure))
        // }
        //
        // let structure = []
        // for (var i = 0; i < in_layer.structure.length; i++) {
        //     let num_neurons = (in_layer.structure[i] - (filter_structure[i] - stride)) / stride
        //     if (!Number.isInteger(num_neurons)) {
        //         let explanation = `(${in_layer.structure[i]} - (${filter_structure[i]} - ${stride})) / ${stride} = ${num_neurons}`
        //         throw new Error(`Incompatible combination of input structure, filter structure, and stride: ${explanation}`)
        //     }
        //
        //     structure.push(num_neurons)
        // }
        //
        // this.createArchitecture()
    }
}

module.exports = ConvolutionalLayer
