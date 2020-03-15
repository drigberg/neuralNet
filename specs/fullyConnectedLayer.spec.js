
const { expect } = require('chai');
const { Net } = require('../lib/net');
const rectifiers = require('../lib/rectifiers');


describe('Fully Connected Layers', () => {
    describe('as input', () => {
        describe('can be created', () => {
            it('with one-dimensional architecture', () => {
                const net = new Net({
                    learning_rate: 0.02,
                    input_architecture: [3],
                    layer_configs: [],
                });

                expect(net.layers[0].states).to.have.length(3);
            });

            it('with two-dimensional architecture', () => {
                const net = new Net({
                    learning_rate: 0.02,
                    input_architecture: [3, 4],
                    layer_configs: [],
                });

                expect(net.layers[0].states).to.have.length(12);
            });

            it('with three-dimensional architecture', () => {
                const net = new Net({
                    learning_rate: 0.02,
                    input_architecture: [3, 4, 5],
                    layer_configs: [],
                });
                expect(net.layers[0].states).to.have.length(60);
            });

            it('with four-dimensional architecture', () => {
                const net = new Net({
                    learning_rate: 0.02,
                    input_architecture: [3, 4, 5, 6],
                    layer_configs: [],
                });
                expect(net.layers[0].states).to.have.length(360);
            });
        });

        describe('cannot be created', () => {
            it('with empty architecture', () => {
                let error_thrown = false;

                try {
                    new Net({
                        input_architecture: [],
                        learning_rate: 0.02,
                        layer_configs: []
                    });
                }
                catch (err) {
                    error_thrown = true;
                    expect(err.message).to.equal('Architecture must be array with at least one dimension');
                }

                expect(error_thrown).to.equal(true);
            });
        });
    });

    describe('as first hidden layer', () => {
        describe('can be created', () => {
            it('with one-dimensional architecture', () => {
                const net = new Net({
                    input_architecture: [3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3],
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(3);
            });

            it('with two-dimensional architecture', () => {
                const net = new Net({
                    input_architecture: [3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3, 4],
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(12);
            });

            it('with three-dimensional architecture', () => {
                const net = new Net({
                    input_architecture: [3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3, 4, 5],
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(60);
            });

            it('with four-dimensional architecture', () => {
                const net = new Net({
                    input_architecture: [3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3, 4, 5, 6],
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(360);
            });
        });

        describe('cannot be created', () => {
            it('with undefined architecture', () => {
                let error_thrown = false;

                try {
                    new Net({
                        input_architecture: [3],
                        learning_rate: 0.02,
                        layer_configs: [
                            {
                                type: 'FULLY_CONNECTED',
                                options: {
                                    architecture: [],
                                    rectifier: rectifiers.relu,
                                }
                            }
                        ]
                    });
                }
                catch (err) {
                    error_thrown = true;
                    expect(err.message).to.equal('Architecture must be array with at least one dimension');
                }

                expect(error_thrown).to.equal(true);
            });
        });
    });

    describe('as further hidden layers', () => {
        describe('can be created', () => {
            it('with one-dimensional architecture', () => {
                const net = new Net({
                    input_architecture: [3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3, 4, 5],
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3],
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(3);
            });

            it('with two-dimensional architecture', () => {
                const net = new Net({
                    input_architecture: [3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3, 4, 5],
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3, 4],
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(12);
            });

            it('with three-dimensional architecture', () => {
                const net = new Net({
                    input_architecture: [3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3, 4, 5],
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3, 4, 5],
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(60);
            });

            it('with four-dimensional architecture', () => {
                const net = new Net({
                    input_architecture: [3],
                    learning_rate: 0.02,
                    layer_configs: [
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3, 4, 5],
                                rectifier: rectifiers.relu,
                            }
                        },
                        {
                            type: 'FULLY_CONNECTED',
                            options: {
                                architecture: [3, 4, 5, 6],
                                rectifier: rectifiers.relu,
                            }
                        }
                    ]
                });

                expect(net.finalLayer.states).to.have.length(360);
            });
        });

        describe('cannot be created', () => {
            it('with undefined architecture', () => {
                let error_thrown = false;

                try {
                    new Net({
                        input_architecture: [3],
                        learning_rate: 0.02,
                        layer_configs: [
                            {
                                type: 'FULLY_CONNECTED',
                                options: {
                                    architecture: [3, 4, 5],
                                    rectifier: rectifiers.relu,
                                }
                            },
                            {
                                type: 'FULLY_CONNECTED',
                                options: {
                                    architecture: [],
                                    rectifier: rectifiers.relu,
                                }
                            }
                        ]
                    });
                }
                catch (err) {
                    error_thrown = true;
                    expect(err.message).to.equal('Architecture must be array with at least one dimension');
                }

                expect(error_thrown).to.equal(true);
            });
        });
    });

    describe('backpropagation:', () => {
        it('all activations are numbers', () => {
            const net = new Net({
                input_architecture: [2],
                learning_rate: 0.02,
                layer_configs: [
                    {
                        type: 'FULLY_CONNECTED',
                        options: {
                            architecture: [3],
                            rectifier: rectifiers.step,
                        }
                    },
                    {
                        type: 'FULLY_CONNECTED',
                        options: {
                            architecture: [2],
                            rectifier: rectifiers.step,
                        }
                    },
                ]
            });

            for (var j = 0; j < 100; j++) {
                net.learn([Math.random(), Math.random()], [0, 1]);
            }


            let all_numbers = true;
            net.layers[0].states.forEach((state) => {
                const activation = net.layers[0].neuronsByState[state].activation;
                if (typeof activation !== 'number') {
                    all_numbers = false;
                }
            });
            expect(all_numbers).to.be.true;
        });

        it('all activations are not zero', () => {
            const net = new Net({
                input_architecture: [2],
                learning_rate: 0.02,
                layer_configs: [
                    {
                        type: 'FULLY_CONNECTED',
                        options: {
                            architecture: [3],
                            rectifier: rectifiers.step,
                        }
                    },
                    {
                        type: 'FULLY_CONNECTED',
                        options: {
                            architecture: [2],
                            rectifier: rectifiers.step,
                        }
                    },
                ]
            });

            for (var j = 0; j < 100; j++) {
                net.learn([Math.random(), Math.random()], [0, 1]);
            }

            let all_zeroes = true;
            net.layers[0].states.forEach((state) => {
                const activation = net.layers[0].neuronsByState[state].activation;
                if (activation !== 0) {
                    all_zeroes = false;
                }
            });
            expect(all_zeroes).to.be.false;
        });
    });
});