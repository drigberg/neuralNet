const FullyConnectedLayer = require("./FullyConnectedLayer")
const ConvolutionalLayer = require("./ConvolutionalLayer")
const fs = require("fs")
const PNG = require('pngjs').PNG

let predictions = 0
let successes = 0
let power = 1

class Net {
    constructor({ architecture, learning_rate }) {
        let input_args = {
            "is_input": true,
            "architecture": architecture,
            "net": this
        }

        let input_layer = new FullyConnectedLayer(input_args)
        this.input_layer = input_layer
        this.layers = []
        this.learning_rate = learning_rate
    }

    get finalLayer() {
        return this.layers.length ? this.layers[this.layers.length - 1] : null
    }

    navigateInput(input) {
        let input_layer = this.input_layer
        nest(input_layer.architecture, input)

        function nest(arch, arr_or_value, index, state) {
            index = index || 0
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    let nested_state = state ? [state, i].join("."): String(i)
                    let nested_arr = arr_or_value[i]
                    let nested = nest(arch, nested_arr, index + 1, nested_state)
                }
            } else {
                input_layer.neurons[state].activation = arr_or_value
                return
            }
        }
    }

    predict(input) {
        if (! (input instanceof Array)) {
            throw new Error("Input must be an array!")
        }

        this.navigateInput(input)

        return this.activate()
    }

    activate() {
        for (var i = 0; i < this.layers.length; i++) {
            this.layers[i].activate()
        }

        return this.finalLayer.activate()
    }

    backPropagate(target) {
        this.finalLayer.propagate(target)

        for (var i = this.layers.length - 2; i >= 0; i--) {
            this.layers[i].propagate()
        }
    }

    learn(input, target) {
        let prediction = this.predict(input)
        let correct = true
        for (var i = 0; i < target.length; i++) {
            if (target[i] !== prediction[i]) {
                correct = false
                break
            }
        }

        predictions += 1
        successes += correct ? 1 : 0

        if (predictions % 256 == 0) {
            power += 1
            console.log(`Prediction #${predictions}:`, prediction, "- Target:", target)
            console.log(`Cumulative ${successes / predictions * 100}% accuracy`)
        }

        this.backPropagate(target)
    }

    addFullyConnectedLayer(layer_args) {
        layer_args = this.supplementLayerArgs(layer_args)

        this.layers.push(new FullyConnectedLayer(layer_args))
    }

    addConvolutionalLayer(layer_args) {
        layer_args = this.supplementLayerArgs(layer_args)

        this.layers.push(new ConvolutionalLayer(layer_args))
    }

    supplementLayerArgs(layer_args) {
        let in_layer = this.finalLayer || this.input_layer

        Object.assign(layer_args, {
            "in_layer": in_layer,
            "net": this
        })

        return layer_args
    }

    loadImageDirectory({ directory }) {
        let promises = []
        let net = this

        return Promise.resolve()
        .then(() => {
            // get models
            return fs.readdirSync(directory)
                .filter((file_name) => {
                    return file_name[0] !== '.';
                })
                .forEach((file_name) => {
                    promises.push(net.loadImage(`${directory}/${file_name}`))
                });
        })
        .then(() => {
            return Promise.all(promises)
        })
        .then((res) => {
            return res
        })
        .catch((err) => {
            console.log("err:", err)
        })
    }

    loadImage(file_path) {
        return new Promise((resolve, reject) => {
            let arr = []
            fs.createReadStream(file_path)
            .pipe(new PNG({ filterType: 4 }))
            .on('parsed', function() {
                console.log(`Parsed ${file_path}!`)
                for (var x = 0; x < this.width; x++) {
                    arr.push([])
                    for (var y = 0; y < this.height; y++) {
                        arr[x].push([])
                        for (var z = 0; z < 3; z++) {
                            let pixel_id = ((this.width * y + x) << 2) + z;
                            let pixel_value = this.data[pixel_id]
                            let state = [x, y, z].join(".")
                            arr[x][y].push(pixel_value)
                        }
                    }
                }
                return resolve(arr)
            })
        })
    }
}

module.exports = Net
