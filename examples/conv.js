const Net = require("../models/Net")
const rectifiers = require("../lib/rectifiers")

net = new Net({
    "architecture": [30, 30, 3],
    "learning_rate": 0.02
})

net.addConvolutionalLayer({
    "filter_structure": [3, 3, 1],
    "depth": 12,
    "stride": 1,
    "rectifier": rectifiers.relu,
})

net.addFullyConnectedLayer({
    "architecture": [3],
    "rectifier": rectifiers.relu,
})