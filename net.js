// partly based on https://github.com/cazala/synaptic/blob/master/src/Neuron.js

const Net = require("./models/Net")
const rectifiers = require("./lib/rectifiers")
const fs = require("fs")

let net = new Net({
    "input_length": 2,
    "learning_rate": 0.01
})

// net.addLayer({
//     "num_neurons": 10,
//     "rectifier": rectifiers.sigmoid,
//     "randomly_disconnected": true
// })


net.addLayer({
    "num_neurons": 40,
    "rectifier": rectifiers.tanh,
})

net.addLayer({
    "num_neurons": 30,
    "rectifier": rectifiers.step,
})

net.addLayer({
    "num_neurons": 1,
    "rectifier": rectifiers.tanh,
})

// let file_data = fs.readFileSync("./data/training.json")
// let data = JSON.parse(file_data)
// let keys = Object.keys(data)

for (var i = 0; i < 5000; i++) {
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


