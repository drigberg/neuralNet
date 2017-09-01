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
            if (! (input instanceof Array)) {
                throw new Error("Input must be an array!")
            }

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

            for (var i = this.layers.length - 2; i >= 0; i--) {
                this.layers[i].propagate()
            }
        })
    }

    learn(input, target) {
        return this.predict(input)
        .then((prediction) => {
            return this.backPropagate(target)
        })
    }

    addLayer(layer_args) {
        let in_layer = this.finalLayer() || this.input

        Object.assign(layer_args, {
            "in_layer": in_layer,
            "net": this
        })

        let new_layer = new Layer(layer_args)
        layer_args.net.layers.push(new_layer)
    }
}

module.exports = Net
