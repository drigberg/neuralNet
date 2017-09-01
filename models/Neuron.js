const Connection = require("./Connection")

let neurons = 0

class Neuron {
    static uuid() {
        return neurons ++
    }

    constructor({ in_layer, layer, randomly_disconnected }) {
        this._id = Neuron.uuid()
        this.layer = layer
        this.randomly_disconnected = randomly_disconnected

        this.connections = {
            "out": {},
            "in": {}
        }

        // create connections to all neurons in input layer
        if (in_layer) {
            this.bias = Math.random() * 0.1 - 0.2

            let sum = 0

            for (var i = 0; i < in_layer.neurons.length; i++) {
                let connect = true
                if (this.randomly_disconnected) {
                    connect = false
                }

                if (connect) {
                    let neuron = in_layer.neurons[i]
                    let weight = Math.random()
                    sum += weight

                    let bias = Math.random() * 0.3
                    let connection = new Connection(neuron, this, weight, bias)

                    // update on both ends of the connection
                    this.connections.in[neuron._id] = connection
                    neuron.connections.out[this._id] = connection
                }
            }

            // ensure that all weights add to 1
            let multiplier = 1 / sum
            let keys = Object.keys(this.connections.in)
            for (var j = 0; j < keys.length; j++) {
                let key = keys[j]
                this.connections.in[key].multiplyBy(multiplier)
            }
        }
    }

    activate() {
        let activation = this.bias
        let connections = this.connections.in
        let keys = Object.keys(connections)

        for (var i = 0; i < keys.length; i++) {
            let connection = connections[keys[i]]
            activation += connection.in_neuron.activation * connection.weight
        }

        this.activation = this.layer.rectifier(activation, false)
        this.derivative = this.layer.rectifier(activation, true)
    }

    propagate(target) {
        if (target !== null) {
            this.error = target - this.activation
        } else {
            let error = 0

            let connections = this.connections.out
            let keys = Object.keys(connections)

            keys.forEach((key) => {
                let connection = connections[key]
                error += connection.out_neuron.error * connection.weight
            })
            this.error = this.derivative * error
        }

        // LEARN
        let keys = Object.keys(this.connections.in)
        for (var j = 0; j < keys.length; j++) {
            let connection = this.connections.in[keys[j]]
            let gradient = this.error * connection.in_neuron.activation
            connection.weight += this.layer.net.learning_rate * gradient
        }

        this.bias += this.layer.net.learning_rate * this.error
    }
}

module.exports = Neuron