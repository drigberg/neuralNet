// partly based on https://github.com/cazala/synaptic/blob/master/src/Neuron.js

/**
 * Module dependencies
 */

const {logger, LEVELS} = require('../lib/logger');
const {Net} = require('../lib/net');
const rectifiers = require('../lib/rectifiers');

/**
 * Module
 */

logger.setLogLevel(LEVELS.DEBUG);

/**
 * Example functionality
 */
function task() {
    const net = new Net({
        inputArchitecture: [2],
        learningRate: 0.02,
        layerConfigs: [
            {
                type: 'FULLY_CONNECTED',
                options: {
                    architecture: [3],
                    rectifier: rectifiers.relu,
                }
            },
            {
                type: 'FULLY_CONNECTED',
                options: {
                    'architecture': [1],
                    'rectifier': rectifiers.step,
                }
            }
        ]
    });
    
    for (let i = 0; i < 100000; i++) {
        const input = [];
        let target = [0];
    
        for (let j = 0; j < 2; j++) {
            const num = Math.random() >= 0.5 ? 1 : 0;
            input.push(num);
        }
    
        if (input[0] + input[1] === 1) {
            target = [1];
        }
    
        net.learn(input, target);
    }    
}

task();
