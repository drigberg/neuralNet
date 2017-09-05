const Net = require("../models/Net")
const rectifiers = require("../lib/rectifiers")

net = new Net({
    "architecture": [30, 30, 3],
    "learning_rate": 0.02
})

net.addConvolutionalLayer({
    "filter_structure": [6, 6, 3],
    "depth": 6,
    "stride": 1,
    "rectifier": rectifiers.relu,
})

net.addFullyConnectedLayer({
    "architecture": [3],
    "rectifier": rectifiers.relu,
})


for (var x = 0; x < 10; x++) {
    let input = []
    let values = []
    let sum = 0

    for (var i = 0; i < 30; i++) {
        input.push([])
        for (var j = 0; j < 30; j++) {
            input[i].push([])
            for (var k = 0; k < 3; k++) {
                let value = Math.random() * 30
                input[i][j].push(Math.floor(value))
                values.push(value)
                sum += value
            }
        }
    }

    console.log(input)

    let average = sum / values.length
    let target = [average > 15, average < 14, average > 16]

    net.learn(input, target)
}
