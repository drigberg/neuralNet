const Neuron = require("./Neuron")

class Layer {
    constructor({ num_neurons, in_layer, bias, rectifier, net, is_input }) {
        this.net = net
        this.neurons = []

        let neuron_args = { "layer": this }

        if (!is_input) {
            this.rectifier = rectifier
            Object.assign(neuron_args, { bias, in_layer })
        }

        for (var i = 0; i < num_neurons; i++) {
            this.neurons.push(new Neuron(neuron_args))
        }
    }

    activate() {
        for (var i = 0; i < this.neurons.length; i++) {
            this.neurons[i].activate()
        }

        return this.getActivations()
    }

    propagate(target_vector) {
        for (var i = 0; i < this.neurons.length; i++) {
            let target = target_vector ? target_vector[i] : null

            this.neurons[i].propagate(target)
        }
    }

    getActivations() {
        return this.neurons.map((neuron) => {
            return neuron.activation
        })
    }
}

module.exports = Layer
