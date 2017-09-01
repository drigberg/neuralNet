// partly based on https://github.com/cazala/synaptic/blob/master/src/Neuron.js

const Net = require("./models/Net")
const rectifiers = require("./lib/rectifiers")
const fs = require("fs")

let bias = 0

let net = new Net({
    "input_length": 3,
    "learning_rate": 0.1
})

net.addLayer({
    "bias": bias,
    "num_neurons": 5,
    "rectifier": rectifiers.step,
})

net.addLayer({
    "bias": bias,
    "num_neurons": 5,
    "rectifier": rectifiers.relu,
})

net.addLayer({
    "bias": bias,
    "num_neurons": 3,
    "rectifier": rectifiers.step,
})

let file_data = fs.readFileSync("./data/training.json")
let data = JSON.parse(file_data)
let keys = Object.keys(data)

for (var i = 0; i < 10000; i++) {
    let input = []
    let target = []

    for (var j = 0; j < 3; j++) {
        let num = Math.random() >= 0.5 ? 1 : 0
        input.push(num)
        target.push(num === 0 ? 1 : 0)
    }

    net.learn(input, target)
    .catch((err) => {
        console.log(err)
    })
}


