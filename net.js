// partly based on https://github.com/cazala/synaptic/blob/master/src/Neuron.js

const Net = require("./models/Net")
const rectifiers = require("./lib/rectifiers")

let bias = -0.5

let net = new Net({
    "input_length": 3,
    "learning_rate": 0.01
})

net.addLayer({
    "bias": bias,
    "num_neurons": 20,
    "rectifier": rectifiers.relu,
})

net.addLayer({
    "bias": bias,
    "num_neurons": 2,
    "rectifier": rectifiers.step,
})

let input = [-100, 50, 3]


net.predict(input)
.then((res) => {
    console.log(res)
})
