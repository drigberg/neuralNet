const FullyConnectedLayer = require("./FullyConnectedLayer")
const ConvolutionalLayer = require("./ConvolutionalLayer")

let predictions = 0
let successes = 0
let power = 1

class Net {
    constructor({ architecture, learning_rate }) {
        let input_args = {
            "is_input": true,
            "architecture": architecture,
            "net": this
        }

        let input_layer = new FullyConnectedLayer(input_args)
        this.input = input_layer
        this.layers = []
        this.learning_rate = learning_rate
    }

    get finalLayer() {
        return this.layers.length ? this.layers[this.layers.length - 1] : null
    }

    predict(input) {
        if (! (input instanceof Array)) {
            throw new Error("Input must be an array!")
        }

        for (var i = 0; i < input.length; i++) {
            this.input.neurons[i].activation = input[i]
        }

        for (var i = 0; i < this.layers.length; i++) {
            this.layers[i].activate()
        }

        return this.finalLayer.activate()
    }

    backPropagate(target) {
        this.finalLayer.propagate(target)

        for (var i = this.layers.length - 2; i >= 0; i--) {
            this.layers[i].propagate()
        }
    }

    learn(input, target) {
        let prediction = this.predict(input)
        let correct = true
        for (var i = 0; i < target.length; i++) {
            if (target[i] !== prediction[i]) {
                correct = false
                break
            }
        }

        predictions += 1
        successes += correct ? 1 : 0

        if (predictions % Math.pow(5, power) == 0) {
            power += 1
            console.log(`Iteration #${predictions}: ${successes / predictions * 100}% accuracy`)
        }

        this.backPropagate(target)
    }

    addFullyConnectedLayer(layer_args) {
        layer_args = this.supplementLayerArgs(layer_args)

        this.layers.push(new FullyConnectedLayer(layer_args))
    }

    addConvolutionalLayer(layer_args) {
        layer_args = this.supplementLayerArgs(layer_args)

        this.layers.push(new ConvolutionalLayer(layer_args))
    }

    supplementLayerArgs(layer_args) {
        let in_layer = this.finalLayer || this.input

        Object.assign(layer_args, {
            "in_layer": in_layer,
            "net": this
        })

        return layer_args
    }
}

module.exports = Net
