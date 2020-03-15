
const { expect } = require('chai');
const {Net} = require('../lib/net');
const rectifiers = require('../lib/rectifiers');
const errors = require('../lib/errors');


describe('Convolutional Layers', () => {
    describe('as first hidden layer', () => {
        describe('can be created', () => {
            it('when filter has same number of dimensions as input layer', () => {
                const net = new Net({
                    input_architecture: [3, 3, 3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [3, 3, 3],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(1);
            });

            it('when filter is smaller than the input layer', () => {
                const net = new Net({
                    input_architecture: [3, 3, 3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [1, 1, 1],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });
                expect(Object.keys(net.finalLayer.neurons)).to.have.length(27);
            });

            it('with multiple filters', () => {
                const net = new Net({
                    input_architecture: [3, 3, 3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [3, 3, 3],
                                depth: 6,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(6);
            });

            it('with stride > 1', () => {
                const net = new Net({
                    input_architecture: [6, 6, 3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [3, 3, 3],
                                depth: 1,
                                stride: 3,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(4);
            });
        });

        describe('cannot be created', () => {
            it('with filter larger than input', () => {
                let error_thrown = false;
                try {
                    new Net({
                        input_architecture: [3, 3, 3],
                        learning_rate: 0.02,
                        layer_configs: [
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filter_architecture: [4, 4, 3],
                                    depth: 1,
                                    stride: 1,
                                    rectifier: rectifiers.relu,
                                }
                            }
                        ]
                    });
                } catch(err) {
                    error_thrown = true;
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER);
                }

                expect(error_thrown).to.be.true;
            });

            it('with stride that pushes filter beyond input', () => {
                let error_thrown = false;

                try {
                    new Net({
                        input_architecture: [3, 3, 3],
                        learning_rate: 0.02,
                        layer_configs: [
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filter_architecture: [2, 2, 2],
                                    depth: 1,
                                    stride: 2,
                                    rectifier: rectifiers.relu,
                                }
                            }
                        ]
                    });
                } catch(err) {
                    error_thrown = true;
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER);
                }

                expect(error_thrown).to.be.true;
            });
        });
    });

    describe('after other convolutional layers', () => {
        describe('can be created', () => {
            it('when filter has same number of dimensions as input layer', () => {
                const net = new Net({
                    input_architecture: [7, 7, 7],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [4, 4, 4],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [3, 3, 3],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(8);
            });

            it('when filter is smaller than the input layer', () => {
                const net = new Net({
                    input_architecture: [7, 7, 7],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [4, 4, 4],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [1, 1, 1],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(64);
            });

            it('with multiple filters', () => {
                const net = new Net({
                    input_architecture: [7, 7, 7],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [4, 4, 4],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [3, 3, 3],
                                depth: 6,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(48);
            });

            it('with stride > 1', () => {
                const net = new Net({
                    input_architecture: [6, 6, 3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [1, 1, 1],
                                depth: 1,
                                stride: 1,
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'CONVOLUTIONAL',
                            options: {
                                filter_architecture: [3, 3, 3],
                                depth: 1,
                                stride: 3,
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(Object.keys(net.finalLayer.neurons)).to.have.length(4);
            });
        });
        describe('cannot be created', () => {
            it('with filter larger than input', () => {
                let error_thrown = false;

                try {
                    new Net({
                        input_architecture: [7, 7, 7],
                        learning_rate: 0.02,
                        layer_configs: [
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filter_architecture: [4, 4, 4],
                                    depth: 1,
                                    stride: 1,
                                    rectifier: rectifiers.relu,
                                }
                            },
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filter_architecture: [5, 5, 3],
                                    depth: 1,
                                    stride: 1,
                                    rectifier: rectifiers.relu,
                                }
                            }
                        ]
                    });
                } catch(err) {
                    error_thrown = true;
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER);
                }
                expect(error_thrown).to.be.true;
            });

            it('with stride that pushes filter beyond input', () => {
                let error_thrown = false;

                try {
                    new Net({
                        input_architecture: [7, 7, 7],
                        learning_rate: 0.02,
                        layer_configs: [
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filter_architecture: [4, 4, 4],
                                    depth: 1,
                                    stride: 1,
                                    rectifier: rectifiers.relu,
                                }
                            },
                            {
                                type: 'CONVOLUTIONAL',
                                options: {
                                    filter_architecture: [2, 2, 2],
                                    depth: 1,
                                    stride: 3,
                                    rectifier: rectifiers.relu,
                                }
                            }
                        ]
                    });
                } catch(err) {
                    error_thrown = true;
                    expect(err.code).to.equal(errors.codes.INCOMPATIBLE_FILTER);
                }

                expect(error_thrown).to.be.true;
            });
        });
    });

    describe('backpropagation:', () => {
        it('all activations are numbers', () => {
            const net = new Net({
                input_architecture: [9, 9, 3],
                learning_rate: 0.0000002,
                layer_configs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filter_architecture: [4, 4, 3],
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

                let all_numbers = true;

                Object.values(net.layers[0].neurons).forEach((neuron) => {
                    if (typeof neuron.activation !== 'number') {
                        all_numbers = false;
                    }
                });

                expect(all_numbers).to.be.true;
            });
        });

        it('all activations are not zero', () => {
            const net = new Net({
                input_architecture: [9, 9, 3],
                learning_rate: 0.0000002,
                layer_configs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filter_architecture: [4, 4, 3],
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

                let all_zeroes = true;

                Object.values(net.layers[0].neurons).forEach((neuron) => {
                    if (neuron.activation !== 0) {
                        all_zeroes = false;
                    }
                });

                expect(all_zeroes).to.be.false;
            });
        });

        it('predictions are numbers', () => {
            const net = new Net({
                input_architecture: [9, 9, 3],
                learning_rate: 0.0000002,
                layer_configs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filter_architecture: [4, 4, 3],
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
                const are_numbers = typeof prediction[0] === 'number' && typeof prediction[1] === 'number';
                const are_not_NaN = (Boolean(prediction[0]) || prediction[0] === 0) &&
                    (Boolean(prediction[1]) || prediction[1] === 0);
                expect(are_numbers).to.be.true;
                expect(are_not_NaN).to.be.true;
            });
        });
    });
});
