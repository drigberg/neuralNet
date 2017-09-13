const Connection = require("./Connection")

let neurons = 0

class Neuron {
    static uuid() {
        return neurons ++
    }

    constructor({ in_neurons, layer, pooling }) {
        this._id = Neuron.uuid()
        this.layer = layer
        this.pooling = pooling
        this.bias = 0

        this.connections = {
            "out": {},
            "in": {}
        }

        // create connections to all neurons in input layer

        if (in_neurons) {
            let sum = 0

            let neuron_keys = Object.keys(in_neurons)
            neuron_keys.forEach((key) => {
                let neuron = in_neurons[key]
                let weight = Math.random()
                sum += weight

                let connection = new Connection({
                    "in_neuron": neuron,
                    "out_neuron": this,
                    "weight": weight
                })

                // update on both ends of the connection
                this.connections.in[neuron._id] = connection
                neuron.connections.out[this._id] = connection
            })

            // ensure that all weights add to 1
            let multiplier = 1 / sum
            let keys = Object.keys(this.connections.in)
            for (var j = 0; j < keys.length; j++) {
                let key = keys[j]
                this.connections.in[key].params.weight *= multiplier
            }
        }
    }

    activate() {
        let activation = this.bias
        let connections = this.connections.in
        let keys = Object.keys(connections)

        if (this.pooling) {
            let max = { "derivative": null, "activation": -Infinity }

            for (var i = 0; i < keys.length; i++) {
                if (connections[keys[i]].in_neuron.activation > max.activation) {
                    max.activation = connections[keys[i]].in_neuron.activation
                    max.derivative = connections[keys[i]].in_neuron.derivative
                }
            }

            this.activation = max.activation
            this.derivative = max.derivative
        } else {
            for (var i = 0; i < keys.length; i++) {
                let connection = connections[keys[i]]
                activation += connection.in_neuron.activation * connection.params.weight
            }

            this.activation = this.layer.rectifier(activation, false)
            this.derivative = this.layer.rectifier(activation, true)
        }
    }

    propagate(target) {
        if (target) {
            this.error = target - this.activation
        } else {
            let error = 0

            let connections = this.connections.out
            let keys = Object.keys(connections)

            keys.forEach((key) => {
                let connection = connections[key]
                error += connection.out_neuron.error * connection.params.weight
            })
            this.error = this.derivative * error
        }

        // LEARN
        let keys = Object.keys(this.connections.in)
        for (var j = 0; j < keys.length; j++) {
            let connection = this.connections.in[keys[j]]
            let gradient = this.error * connection.in_neuron.activation
            connection.params.weight += this.layer.net.learning_rate * gradient
        }

        this.bias += this.layer.net.learning_rate * this.error
    }
}

module.exports = Neuron
