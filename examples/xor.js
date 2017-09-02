// partly based on https://github.com/cazala/synaptic/blob/master/src/Neuron.js

const Net = require("../models/Net")
const rectifiers = require("../lib/rectifiers")

const net = new Net({
    "input_length": 2,
    "learning_rate": 0.05
})

net.addLayer({
    "num_neurons": 3,
    "rectifier": rectifiers.relu
})

net.addLayer({
    "num_neurons": 1,
    "rectifier": rectifiers.step,
})

for (var i = 0; i < 80000; i++) {
    let input = []
    let target = [0]

    for (var j = 0; j < 2; j++) {
        let num = Math.random() >= 0.5 ? 1 : 0
        input.push(num)
    }

    if (input[0] + input[1] === 1) {
        target = [1]
    }

    net.learn(input, target)
}
