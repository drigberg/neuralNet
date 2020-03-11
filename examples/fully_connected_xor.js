// partly based on https://github.com/cazala/synaptic/blob/master/src/Neuron.js

/**
 * Module dependencies
 */

const {Net} = require('../models/Net');
const rectifiers = require('../lib/rectifiers');

/**
 * Module
 */

function task() {
    const net = new Net({
        'architecture': [2],
        'learning_rate': 0.02
    });
    
    net.addFullyConnectedLayer({
        'architecture': [3],
        'rectifier': rectifiers.relu,
    });
    
    net.addFullyConnectedLayer({
        'architecture': [1],
        'rectifier': rectifiers.step,
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