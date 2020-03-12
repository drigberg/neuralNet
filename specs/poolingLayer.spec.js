

const { expect } = require('chai');
const { Net } = require('../lib/net');
const rectifiers = require('../lib/rectifiers');


describe('Pooling Layers', () => {
    describe('can be created', () => {
        it('when spatial extent is compatible with input architecture', () => {
            const net = new Net({
                input_architecture: [7, 7, 7],
                learning_rate: 0.0000002,
                layer_configs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filter_structure: [4, 4, 4],
                            depth: 1,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'POOLING',
                        options: {
                            spatial_extent: 2
                        }
                    }
                ]
            });

            expect(Object.keys(net.finalLayer.neurons)).to.have.length(16);
        });
    });

    describe('backpropagation:', () => {
        it('all activations are numbers', () => {
            const net = new Net({
                input_architecture: [9, 9, 3],
                learning_rate: 0.000000002,
                layer_configs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filter_structure: [4, 4, 3],
                            depth: 2,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'POOLING',
                        options: {
                            spatial_extent: 2
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

            return net.loadImage(__dirname + '/data/gabri_size_9.png')
                .then((image) => {
                    net.learn(image, [0, 1]);

                    const pooling_neurons = net.layers[1].neurons;
                    const neuron_keys = Object.keys(pooling_neurons);
                    let all_numbers = true;

                    for (var i = 0; i < neuron_keys.length; i++) {
                        const activation = pooling_neurons[neuron_keys[i]].activation;
                        if (typeof activation !== 'number') {
                            all_numbers = false;
                            break;
                        }
                    }

                    expect(all_numbers).to.be.true;
                });
        });

        it('all activations are not zero', () => {
            const net = new Net({
                input_architecture: [9, 9, 3],
                learning_rate: 0.000000002,
                layer_configs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filter_structure: [4, 4, 3],
                            depth: 2,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'POOLING',
                        options: {
                            spatial_extent: 2
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

            return net.loadImage(__dirname + '/data/gabri_size_9.png')
                .then((image) => {
                    for (var j = 0; j < 100; j++) {
                        net.learn(image, [Math.random(), Math.random()]);
                    }

                    const pooling_neurons = net.layers[1].neurons;
                    const neuron_keys = Object.keys(pooling_neurons);
                    let all_zeroes = true;

                    for (var i = 0; i < neuron_keys.length; i++) {
                        const activation = pooling_neurons[neuron_keys[i]].activation;

                        if (activation !== 0) {
                            all_zeroes = false;
                            break;
                        }
                    }

                    expect(all_zeroes).to.be.false;
                });
        });

        it('predictions are numbers', () => {
            const net = new Net({
                input_architecture: [9, 9, 3],
                learning_rate: 0.000000002,
                layer_configs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filter_structure: [4, 4, 3],
                            depth: 2,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'POOLING',
                        options: {
                            spatial_extent: 2
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

            return net.loadImage(__dirname + '/data/gabri_size_9.png')
                .then((image) => {
                    for (var j = 0; j < 100; j++) {
                        net.learn(image, [Math.random(), Math.random()]);
                    }

                    const prediction = net.predict(image, [Math.random(), Math.random()]);
                    const are_numbers = typeof prediction[0] === 'number' && typeof prediction[1] === 'number';
                    const are_not_NaN = (Boolean(prediction[0]) || prediction[0] === 0) &&
                        (Boolean(prediction[1]) || prediction[1] === 0);
                    expect(are_numbers).to.be.true;
                    expect(are_not_NaN).to.be.true;
                });
        });

        it('[This scenario fails! Figure out why!]', () => {
            new Net({
                input_architecture: [9, 9, 3],
                learning_rate: 0.000000002,
                layer_configs: [
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filter_structure: [4, 4, 3],
                            depth: 2,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'CONVOLUTIONAL',
                        options: {
                            filter_structure: [4, 4, 3],
                            depth: 2,
                            stride: 1,
                            rectifier: rectifiers.relu,
                        }
                    },
                    {
                        type: 'POOLING',
                        options: {
                            spatial_extent: 3
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
