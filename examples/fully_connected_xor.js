// partly based on https://github.com/cazala/synaptic/blob/master/src/Neuron.js

/**
 * Module dependencies
 */

const {Net} = require('../lib/net');
const rectifiers = require('../lib/rectifiers');

/**
 * Module
 */

/**
 * Example functionality
 */
function task() {
    const net = new Net({
        input_architecture: [2],
        learning_rate: 0.02,
        layer_configs: [
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
    
    for (var i = 0; i < 80000; i++) {
        const input = [];
        let target = [0];
    
        for (var j = 0; j < 2; j++) {
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
