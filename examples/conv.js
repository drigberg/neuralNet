const Net = require("../models/Net")

const net = new Net({
    "input_structure": [32, 32, 3],
    "learning_rate": 0.02
})

net.addConvolutionalLayer({
    "filter_structure": [3, 3, 3],
    "depth": 6,
    "stride": 1,
    "rectifier": rectifiers.relu,
})
