const Layer = require("./Layer")

class Net {
    constructor({ input_length, learning_rate }) {
        let input_args = {
            "is_input": true,
            "num_neurons": input_length,
            "net": this
        }

        let input_layer = new Layer(input_args)
        this.input = input_layer
        this.layers = []
        this.learning_rate = learning_rate
    }

    finalLayer() {
        return this.layers.length ? this.layers[this.layers.length - 1] : null
    }

    predict(input) {
        return new Promise((resolve) => {
            for (var i = 0; i < input.length; i++) {
                this.input.neurons[i].activation = input[i]
            }

            for (var i = 0; i < this.layers.length; i++) {
                this.layers[i].activate()
            }

            return resolve(this.finalLayer().activate())
        })
    }

    backPropagate(target) {
        return new Promise((resolve) => {
            this.finalLayer().propagate(target)

            for (var i = input.length - 2; i >= 0; i--) {
                this.layers[i].propagate()
            }
        })
    }

    learn(input) {
        return this.predict(input)
        .then(() => {
            return this.backPropagate()
        })
    }

    addLayer(layer_args) {
        Object.assign(layer_args, {
            "in_layer": this.finalLayer() || this.input,
            "net": this
        })

        let new_layer = new Layer(layer_args)
        layer_args.net.layers.push(new_layer)
    }
}

module.exports = Net
