const Net = require("../models/Net")
const rectifiers = require("../lib/rectifiers")

const net = new Net({
    "architecture": [32, 32, 3],
    "learning_rate": 0.02
})

// const jedi_images = net.loadImageDirectory({"directory": "./data/training/jedi"})
// const sith_images = net.loadImageDirectory({"directory": "./data/training/sith"})


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

net.loadImageDirectory({"directory": "./data/photos"})
.then((darth_images) => {
    for (var i = 0; i < 100; i++) {
        net.learn(darth_images[0], [0, 0, 1])
    }
})


// let targets = {
//     "jedi": [1, 0],
//     "sith": [0, 1]
// }

// for (var i = 0; i < Math.max(jedi_images.length, sith_images.length); i++) {
//     if (i < jedi_images.length) {
//         net.learn(jedi_images[i], targets.jedi)
//     }

//     if (i < sith_images.length) {
//         net.learn(sith_images[i], targets.sith)
//     }
// }