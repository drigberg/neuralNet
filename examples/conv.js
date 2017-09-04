const Net = require("../models/Net")
const rectifiers = require("../lib/rectifiers")

const net = new Net({
    "architecture": [9, 3, 3],
    "learning_rate": 0.02
})

net.addConvolutionalLayer({
    "filter_structure": [3, 1, 1],
    "depth": 1,
    "stride": 1,
    "rectifier": rectifiers.relu,
})
