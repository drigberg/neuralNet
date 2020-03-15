
const { expect } = require('chai');
const { Net } = require('../lib/net');
const rectifiers = require('../lib/rectifiers');


describe('Fully Connected Layers', () => {
    describe('as input', () => {
        describe('can be created', () => {
            it('with one-dimensional architecture', () => {
                const net = new Net({
                    learningRate: 0.02,
                    inputArchitecture: [3],
                    layerConfigs: [],
                });

                expect(net.inputLayer.states).to.have.length(3);
            });

            it('with two-dimensional architecture', () => {
                const net = new Net({
                    learningRate: 0.02,
                    inputArchitecture: [3, 4],
                    layerConfigs: [],
                });

                expect(net.inputLayer.states).to.have.length(12);
            });

            it('with three-dimensional architecture', () => {
                const net = new Net({
                    learningRate: 0.02,
                    inputArchitecture: [3, 4, 5],
                    layerConfigs: [],
                });
                expect(net.inputLayer.states).to.have.length(60);
            });

            it('with four-dimensional architecture', () => {
                const net = new Net({
                    learningRate: 0.02,
                    inputArchitecture: [3, 4, 5, 6],
                    layerConfigs: [],
                });
                expect(net.inputLayer.states).to.have.length(360);
            });
        });

        describe('cannot be created', () => {
            it('with empty architecture', () => {
                let errorThrown = false;

                try {
                    new Net({
                        inputArchitecture: [],
                        learningRate: 0.02,
                        layerConfigs: []
                    });
                }
                catch (err) {
                    errorThrown = true;
                    expect(err.message).to.equal('Architecture must be array with at least one dimension');
                }

                expect(errorThrown).to.equal(true);
            });
        });
    });

    describe('as first hidden layer', () => {
        describe('can be created', () => {
            it('with one-dimensional architecture', () => {
                const net = new Net({
                    inputArchitecture: [3],
                    learningRate: 0.02,
                    layerConfigs: [
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
                    inputArchitecture: [3],
                    learningRate: 0.02,
                    layerConfigs: [
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
                    inputArchitecture: [3],
                    learningRate: 0.02,
                    layerConfigs: [
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
                    inputArchitecture: [3],
                    learningRate: 0.02,
                    layerConfigs: [
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
                let errorThrown = false;

                try {
                    new Net({
                        inputArchitecture: [3],
                        learningRate: 0.02,
                        layerConfigs: [
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
                    errorThrown = true;
                    expect(err.message).to.equal('Architecture must be array with at least one dimension');
                }

                expect(errorThrown).to.equal(true);
            });
        });
    });

    describe('as further hidden layers', () => {
        describe('can be created', () => {
            it('with one-dimensional architecture', () => {
                const net = new Net({
                    inputArchitecture: [3],
                    learningRate: 0.02,
                    layerConfigs: [
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
                    inputArchitecture: [3],
                    learningRate: 0.02,
                    layerConfigs: [
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
                    inputArchitecture: [3],
                    learningRate: 0.02,
                    layerConfigs: [
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
                    inputArchitecture: [3],
                    learningRate: 0.02,
                    layerConfigs: [
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
                let errorThrown = false;

                try {
                    new Net({
                        inputArchitecture: [3],
                        learningRate: 0.02,
                        layerConfigs: [
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
                    errorThrown = true;
                    expect(err.message).to.equal('Architecture must be array with at least one dimension');
                }

                expect(errorThrown).to.equal(true);
            });
        });
    });

    describe('backpropagation:', () => {
        it('all activations are numbers', () => {
            const net = new Net({
                inputArchitecture: [2],
                learningRate: 0.02,
                layerConfigs: [
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


            let allNumbers = true;
            net.inputLayer.states.forEach((state) => {
                const activation = net.inputLayer.neuronsByState[state].activation;
                if (typeof activation !== 'number') {
                    allNumbers = false;
                }
            });
            expect(allNumbers).to.be.true;
        });

        it('all activations are not zero', () => {
            const net = new Net({
                inputArchitecture: [2],
                learningRate: 0.02,
                layerConfigs: [
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

            let allZeroes = true;
            net.inputLayer.states.forEach((state) => {
                const activation = net.inputLayer.neuronsByState[state].activation;
                if (activation !== 0) {
                    allZeroes = false;
                }
            });
            expect(allZeroes).to.be.false;
        });
    });
});