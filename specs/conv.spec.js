
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
            it.only("when filter has same number of dimensions as input layer", () => {
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

            it("with > 50000 neurons", () => {
                net = new Net({
                    "architecture": [96, 96, 3],
                    "learning_rate": 0.02
                })

                net.addConvolutionalLayer({
                    "filter_structure": [6, 6, 3],
                    "depth": 7,
                    "stride": 1,
                    "rectifier": rectifiers.relu,
                })

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(57967)
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
})