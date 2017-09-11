const Net = require("../models/Net")
const rectifiers = require("../lib/rectifiers")

const net = new Net({
    "architecture": [9, 9, 3],
    "learning_rate": 0.000001
})

net.addConvolutionalLayer({
    "filter_structure": [3, 3, 1],
    "depth": 2,
    "stride": 1,
    "rectifier": rectifiers.relu,
})

net.addFullyConnectedLayer({
    "architecture": [2],
    "rectifier": rectifiers.step,
})

let targets = {
    "jedi": [1, 0],
    "sith": [0, 1]
}

let train_promises = [
    net.loadImageDirectory({"directory": "./data/training_9/jedi"}),
    net.loadImageDirectory({"directory": "./data/training_9/sith"})
]

let test_promises = [
    net.loadImageDirectory({"directory": "./data/testing_9/sith"}),
    net.loadImageDirectory({"directory": "./data/testing_9/jedi"}),
    net.loadImageDirectory({"directory": "./data/testing_9/gabri"}),
    net.loadImageDirectory({"directory": "./data/testing_9/jj"}),
    net.loadImageDirectory({"directory": "./data/testing_9/konstantin"})
]

let power = 1

Promise.all(train_promises)
.then(([jedi_images, sith_images]) => {
    return Promise.all(test_promises)
    .then(([sith, jedi, gabri, jj, konstantin]) => {
        for (var i = 0; i < 500000; i++) {
            if (Math.random() < 0.5) {
                let jedi_index = Math.floor(Math.random() * jedi_images.length)
                net.learn(jedi_images[jedi_index], targets.jedi)
            } else {
                let sith_index = Math.floor(Math.random() * sith_images.length)
                net.learn(sith_images[sith_index], targets.sith)
            }

            if (i % 256 == 0) {
                power += 1
                test(sith, jedi, gabri, jj, konstantin)
            }
        }
    })
})

function test(sith, jedi, gabri, jj, konstantin) {
    sith.forEach((test_image) => {
        let prediction = net.predict(test_image)
        console.log("Test prediction for sith:", prediction)
    })

    jedi.forEach((test_image) => {
        let prediction = net.predict(test_image)
        console.log("Test prediction for jedi:", prediction)
    })

    jj.forEach((test_image) => {
        let prediction = net.predict(test_image)
        console.log("Test prediction for jj:", prediction)
    })

    konstantin.forEach((test_image) => {
        let prediction = net.predict(test_image)
        console.log("Test prediction for konstantin:", prediction)
    })

    gabri.forEach((test_image) => {
        let prediction = net.predict(test_image)
        console.log("Test prediction for gabri:", prediction)
    })
}
