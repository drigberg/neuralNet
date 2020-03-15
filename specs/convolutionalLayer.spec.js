
const { expect } = require('chai');
const {Net} = require('../lib/net');
const rectifiers = require('../lib/rectifiers');
const errors = require('../lib/errors');


describe('Convolutional Layers', () => {
    describe('as first hidden layer', () => {
        describe('can be created', () => {
            it('when filter has same number of dimensions as input layer', () => {
                const net = new Net({
                    inputArchitecture: [3, 3, 3],
                    learningRate: 0.02,
                    layerConfigs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [3, 3, 3],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(1);
            });

            it('when filter is smaller than the input layer', () => {
                const net = new Net({
                    inputArchitecture: [3, 3, 3],
                    learningRate: 0.02,
                    layerConfigs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [1, 1, 1],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });
                expect(net.finalLayer.states).to.have.length(27);
            });

            it('with multiple filters', () => {
                const net = new Net({
                    inputArchitecture: [3, 3, 3],
                    learningRate: 0.02,
                    layerConfigs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [3, 3, 3],
                                depth: 6,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(6);
            });

            it('with stride > 1', () => {
                const net = new Net({
                    inputArchitecture: [6, 6, 3],
                    learningRate: 0.02,
                    layerConfigs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [3, 3, 3],
                                depth: 1,
                                stride: 3,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(4);
            });
        });

        describe('cannot be created', () => {
            it('with filter larger than input', () => {
                let errorThrown = false;
                try {
                    new Net({
                        inputArchitecture: [3, 3, 3],
                        learningRate: 0.02,
                        layerConfigs: [
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filterArchitecture: [4, 4, 3],
                                    depth: 1,
                                    stride: 1,
                                    rectifier: rectifiers.relu,
                                }
                            }
                        ]
                    });
                } catch(err) {
                    errorThrown = true;
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER);
                }

                expect(errorThrown).to.be.true;
            });

            it('with stride that pushes filter beyond input', () => {
                let errorThrown = false;

                try {
                    new Net({
                        inputArchitecture: [3, 3, 3],
                        learningRate: 0.02,
                        layerConfigs: [
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filterArchitecture: [2, 2, 2],
                                    depth: 1,
                                    stride: 2,
                                    rectifier: rectifiers.relu,
                                }
                            }
                        ]
                    });
                } catch(err) {
                    errorThrown = true;
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER);
                }

                expect(errorThrown).to.be.true;
            });
        });
    });

    describe('after other convolutional layers', () => {
        describe('can be created', () => {
            it('when filter has same number of dimensions as input layer', () => {
                const net = new Net({
                    inputArchitecture: [7, 7, 7],
                    learningRate: 0.02,
                    layerConfigs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [4, 4, 4],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [3, 3, 3],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(8);
            });

            it('when filter is smaller than the input layer', () => {
                const net = new Net({
                    inputArchitecture: [7, 7, 7],
                    learningRate: 0.02,
                    layerConfigs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [4, 4, 4],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [1, 1, 1],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(64);
            });

            it('with multiple filters', () => {
                const net = new Net({
                    inputArchitecture: [7, 7, 7],
                    learningRate: 0.02,
                    layerConfigs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [4, 4, 4],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [3, 3, 3],
                                depth: 6,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(48);
            });

            it('with stride > 1', () => {
                const net = new Net({
                    inputArchitecture: [6, 6, 3],
                    learningRate: 0.02,
                    layerConfigs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [1, 1, 1],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filterArchitecture: [3, 3, 3],
                                depth: 1,
                                stride: 3,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(4);
            });
        });
        describe('cannot be created', () => {
            it('with filter larger than input', () => {
                let errorThrown = false;

                try {
                    new Net({
                        inputArchitecture: [7, 7, 7],
                        learningRate: 0.02,
                        layerConfigs: [
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filterArchitecture: [4, 4, 4],
                                    depth: 1,
                                    stride: 1,
                                    rectifier: rectifiers.relu,
                                }
                            },
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filterArchitecture: [5, 5, 3],
                                    depth: 1,
                                    stride: 1,
                                    rectifier: rectifiers.relu,
                                }
                            }
                        ]
                    });
                } catch(err) {
                    errorThrown = true;
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER);
                }
                expect(errorThrown).to.be.true;
            });

            it('with stride that pushes filter beyond input', () => {
                let errorThrown = false;

                try {
                    new Net({
                        inputArchitecture: [7, 7, 7],
                        learningRate: 0.02,
                        layerConfigs: [
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filterArchitecture: [4, 4, 4],
                                    depth: 1,
                                    stride: 1,
                                    rectifier: rectifiers.relu,
                                }
                            },
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filterArchitecture: [2, 2, 2],
                                    depth: 1,
                                    stride: 3,
                                    rectifier: rectifiers.relu,
                                }
                            }
                        ]
                    });
                } catch(err) {
                    errorThrown = true;
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER);
                }

                expect(errorThrown).to.be.true;
            });
        });
    });

    describe('backpropagation:', () => {
        it('all activations are numbers', () => {
            const net = new Net({
                inputArchitecture: [9, 9, 3],
                learningRate: 0.0000002,
                layerConfigs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filterArchitecture: [4, 4, 3],
                            depth: 3,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'FULLY_CONNECTED',
                        options: {
                            architecture: [2],
                            rectifier: rectifiers.identity,
                        }
                    }
                ]
            });

            return net.loadImage(__dirname + '/data/9x9x3.png')
            .then((image) => {
                net.learn(image, [0, 1]);

                let allNumbers = true;

                Object.values(net.inputLayer.neuronsByState).forEach((neuron) => {
                    if (typeof neuron.activation !== 'number') {
                        allNumbers = false;
                    }
                });

                expect(allNumbers).to.be.true;
            });
        });

        it('all activations are not zero', () => {
            const net = new Net({
                inputArchitecture: [9, 9, 3],
                learningRate: 0.0000002,
                layerConfigs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filterArchitecture: [4, 4, 3],
                            depth: 3,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'FULLY_CONNECTED',
                        options: {
                            architecture: [2],
                            rectifier: rectifiers.identity,
                        }
                    }
                ]
            });

            return net.loadImage(__dirname + '/data/9x9x3.png')
            .then((image) => {
                for (var j = 0; j < 100; j++) {
                    net.learn(image, [Math.random(), Math.random()]);
                }

                let allZeroes = true;

                Object.values(net.inputLayer.neuronsByState).forEach((neuron) => {
                    if (neuron.activation !== 0) {
                        allZeroes = false;
                    }
                });

                expect(allZeroes).to.be.false;
            });
        });

        it('predictions are numbers', () => {
            const net = new Net({
                inputArchitecture: [9, 9, 3],
                learningRate: 0.0000002,
                layerConfigs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filterArchitecture: [4, 4, 3],
                            depth: 3,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'FULLY_CONNECTED',
                        options: {
                            architecture: [2],
                            rectifier: rectifiers.identity,
                        }
                    }
                ]
            });

            return net.loadImage(__dirname + '/data/9x9x3.png')
            .then((image) => {
                for (var j = 0; j < 100; j++) {
                    net.learn(image, [Math.random(), Math.random()]);
                }

                const prediction = net.predict(image, [Math.random(), Math.random()]);
                console.log(prediction);
                const areNumbers = typeof prediction[0] === 'number' && typeof prediction[1] === 'number';
                const areNotNaN = (Boolean(prediction[0]) || prediction[0] === 0) &&
                    (Boolean(prediction[1]) || prediction[1] === 0);
                expect(areNumbers).to.be.true;
                expect(areNotNaN).to.be.true;
            });
        });
    });
});
