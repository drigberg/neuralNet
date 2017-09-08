
const { expect } = require("chai")
const Neuron = require("../models/Neuron")
const FullyConnectedLayer = require("../models/FullyConnectedLayer")
const Net = require("../models/Net")
const rectifiers = require("../lib/rectifiers")
const errors = require("../lib/errors")


describe("Convolutional Layers", () => {
    describe("as first hidden layer", () => {
        let net

        beforeEach(() => {
            net = new Net({
                "architecture": [3, 3, 3],
                "learning_rate": 0.02
            })
        })

        describe("can be created", () => {
            it("when filter has same number of dimensions as input layer", () => {
                net.addConvolutionalLayer({
                    "filter_structure": [3, 3, 3],
                    "depth": 1,
                    "stride": 1,
                    "rectifier": rectifiers.relu,
                })

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(1)
            })

            it("when filter is smaller than the input layer", () => {
                net.addConvolutionalLayer({
                    "filter_structure": [1, 1, 1],
                    "depth": 1,
                    "stride": 1,
                    "rectifier": rectifiers.relu,
                })

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(27)
            })

            it("with multiple filters", () => {
                net.addConvolutionalLayer({
                    "filter_structure": [3, 3, 3],
                    "depth": 6,
                    "stride": 1,
                    "rectifier": rectifiers.relu,
                })

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(6)
            })

            it("with stride > 1", () => {
                net = new Net({
                    "architecture": [6, 6, 3],
                    "learning_rate": 0.02
                })

                net.addConvolutionalLayer({
                    "filter_structure": [3, 3, 3],
                    "depth": 1,
                    "stride": 3,
                    "rectifier": rectifiers.relu,
                })

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(4)
            })
        })

        describe("cannot be created", () => {
            it("with filter larger than input", () => {
                let error_thrown = false

                try {
                    net.addConvolutionalLayer({
                        "filter_structure": [4, 4, 3],
                        "depth": 1,
                        "stride": 1,
                        "rectifier": rectifiers.relu,
                    })
                } catch(err) {
                    error_thrown = true
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER)
                }

                expect(error_thrown).to.be.true
            })

            it("with stride that pushes filter beyond input", () => {
                let error_thrown = false

                try {
                    net.addConvolutionalLayer({
                        "filter_structure": [2, 2, 2],
                        "depth": 1,
                        "stride": 2,
                        "rectifier": rectifiers.relu,
                    })
                } catch(err) {
                    error_thrown = true
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER)
                }

                expect(error_thrown).to.be.true
            })
        })
    })

    describe("after other convolutional layers", () => {
        let net

        beforeEach(() => {
            net = new Net({
                "architecture": [7, 7, 7],
                "learning_rate": 0.02
            })
            net.addConvolutionalLayer({
                "filter_structure": [4, 4, 4],
                "depth": 1,
                "stride": 1,
                "rectifier": rectifiers.relu,
            })
        })

        describe("can be created", () => {
            it("when filter has same number of dimensions as input layer", () => {
                net.addConvolutionalLayer({
                    "filter_structure": [3, 3, 3],
                    "depth": 1,
                    "stride": 1,
                    "rectifier": rectifiers.relu,
                })

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(8)
            })

            it("when filter is smaller than the input layer", () => {
                net.addConvolutionalLayer({
                    "filter_structure": [1, 1, 1],
                    "depth": 1,
                    "stride": 1,
                    "rectifier": rectifiers.relu,
                })

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(64)
            })

            it("with multiple filters", () => {
                net.addConvolutionalLayer({
                    "filter_structure": [3, 3, 3],
                    "depth": 6,
                    "stride": 1,
                    "rectifier": rectifiers.relu,
                })

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(48)
            })

            it("with stride > 1", () => {
                net = new Net({
                    "architecture": [6, 6, 3],
                    "learning_rate": 0.02
                })

                net.addConvolutionalLayer({
                    "filter_structure": [1, 1, 1],
                    "depth": 1,
                    "stride": 1,
                    "rectifier": rectifiers.relu,
                })

                net.addConvolutionalLayer({
                    "filter_structure": [3, 3, 3],
                    "depth": 1,
                    "stride": 3,
                    "rectifier": rectifiers.relu,
                })

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(4)
            })
        })
        describe("cannot be created", () => {
            it("with filter larger than input", () => {
                let error_thrown = false

                try {
                    net.addConvolutionalLayer({
                        "filter_structure": [5, 5, 3],
                        "depth": 1,
                        "stride": 1,
                        "rectifier": rectifiers.relu,
                    })
                } catch(err) {
                    error_thrown = true
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER)
                }

                expect(error_thrown).to.be.true
            })

            it("with stride that pushes filter beyond input", () => {
                let error_thrown = false

                try {
                    net.addConvolutionalLayer({
                        "filter_structure": [2, 2, 2],
                        "depth": 1,
                        "stride": 3,
                        "rectifier": rectifiers.relu,
                    })
                } catch(err) {
                    error_thrown = true
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER)
                }

                expect(error_thrown).to.be.true
            })
        })
    })

    describe.only("backpropagation:", () => {
        let net

        beforeEach(() => {
            net = new Net({
                "architecture": [9, 9, 3],
                "learning_rate": 0.0002
            })

            net.addConvolutionalLayer({
                "filter_structure": [4, 4, 3],
                "depth": 3,
                "stride": 1,
                "rectifier": rectifiers.relu,
            })

            net.addFullyConnectedLayer({
                "architecture": [2],
                "rectifier": rectifiers.identity,
            })
        })

        it("all activations are numbers", () => {
            return net.loadImage(__dirname + "/data/gabri_size_9.png")
            .then((image) => {
                net.learn(image, [0, 1])

                let conv_neurons = net.layers[0].neurons
                let neuron_keys = Object.keys(conv_neurons)
                let all_numbers = true

                for (var i = 0; i < neuron_keys.length; i++) {
                    let activation = conv_neurons[neuron_keys[i]].activation
                    if (typeof activation !== "number") {
                        all_numbers = false
                        break
                    }
                }

                expect(all_numbers).to.be.true
            })
        })

        it("all activations are not zero", () => {
            return net.loadImage(__dirname + "/data/gabri_size_9.png")
            .then((image) => {
                for (var j = 0; j < 100; j++) {
                    net.learn(image, [Math.random(), Math.random()])
                }

                let conv_neurons = net.layers[0].neurons
                let neuron_keys = Object.keys(conv_neurons)
                let all_zeroes = true

                for (var i = 0; i < neuron_keys.length; i++) {
                    let activation = conv_neurons[neuron_keys[i]].activation

                    if (activation !== 0) {
                        all_zeroes = false
                        break
                    }
                }

                expect(all_zeroes).to.be.false
            })
        })

        it("predictions are numbers", () => {
            return net.loadImage(__dirname + "/data/gabri_size_9.png")
            .then((image) => {
                for (var j = 0; j < 100; j++) {
                    net.learn(image, [Math.random(), Math.random()])
                }

                let prediction = net.predict(image, [Math.random(), Math.random()])
                console.log(prediction)
                let are_numbers = typeof prediction[0] === "number" && typeof prediction[1] === "number"
                let are_not_NaN = (Boolean(prediction[0]) || prediction[0] === 0) &&
                    (Boolean(prediction[1]) || prediction[1] === 0)
                expect(are_numbers).to.be.true
                expect(are_not_NaN).to.be.true
            })
        })
    })
})