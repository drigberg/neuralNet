

const { expect } = require('chai');
const { Net } = require('../lib/net');
const rectifiers = require('../lib/rectifiers');


describe('Pooling Layers', () => {
    describe('can be created', () => {
        it('when spatial extent is compatible with input architecture', () => {
            const net = new Net({
                inputArchitecture: [7, 7, 7],
                learningRate: 0.0000002,
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
                        type: 'POOLING',
                        options: {
                            spatialExtent: 2
                        }
                    }
                ]
            });

            expect(net.finalLayer.states).to.have.length(16);
        });
    });

    describe('backpropagation:', () => {
        it('all activations are numbers', () => {
            const net = new Net({
                inputArchitecture: [9, 9, 3],
                learningRate: 0.000000002,
                layerConfigs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filterArchitecture: [4, 4, 3],
                            depth: 2,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'POOLING',
                        options: {
                            spatialExtent: 2
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
                    Object.values(net.layers[1].neuronsByState).forEach((neuron) => {
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
                learningRate: 0.000000002,
                layerConfigs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filterArchitecture: [4, 4, 3],
                            depth: 2,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'POOLING',
                        options: {
                            spatialExtent: 2
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
                    Object.values(net.layers[1].neuronsByState).forEach((neuron) => {
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
                learningRate: 0.000000002,
                layerConfigs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filterArchitecture: [4, 4, 3],
                            depth: 2,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'POOLING',
                        options: {
                            spatialExtent: 2
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
                    const areNumbers = typeof prediction[0] === 'number' && typeof prediction[1] === 'number';
                    const areNotNaN = (Boolean(prediction[0]) || prediction[0] === 0) &&
                        (Boolean(prediction[1]) || prediction[1] === 0);
                    expect(areNumbers).to.be.true;
                    expect(areNotNaN).to.be.true;
                });
        });

        it('[This scenario fails! Figure out why!]', () => {
            new Net({
                inputArchitecture: [9, 9, 3],
                learningRate: 0.000000002,
                layerConfigs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filterArchitecture: [4, 4, 3],
                            depth: 2,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filterArchitecture: [4, 4, 1],
                            depth: 1,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'POOLING',
                        options: {
                            spatialExtent: 3
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
        });
    });
});
