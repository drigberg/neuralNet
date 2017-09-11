const Neuron = require("./Neuron")
const _ = require("lodash")

/**
 * Layers contain neurons and expose functions for accessing them
 *
 * @class
 * @param {Number} net - the net to which the layer belongs
 * @param {Number} in_layer - layer forming incoming connections
 * @param {Number} rectifier - activation function
 */
class Layer {
    constructor({ net, in_layer, is_input, rectifier }) {
        this.net = net
        this.in_layer = in_layer

        // rectifier is only used for non-input layers
        this.rectifier = is_input ? null : rectifier
    }

    activate() {
        for (var i = 0; i < this.neuronsAsArray.length; i++) {
            this.neuronsAsArray[i].activate()
        }

        return this.getActivations()
    }

    propagate(target_vector) {
        for (var i = 0; i < this.neuronsAsArray.length; i++) {
            let target = target_vector ? target_vector[i] : null
            this.neuronsAsArray[i].propagate(target)
        }
    }

    getActivations() {
        return this.neuronsAsArray.map((neuron) => {
            return neuron.activation
        })
    }

    get neuronsAsArray() {
        return _.values(this.neurons)
    }
}

module.exports = Layer
