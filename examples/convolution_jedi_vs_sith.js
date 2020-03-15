/**
 * Module dependencies
 */

const {Net} = require('../models/Net');
const rectifiers = require('../lib/rectifiers');

/**
 * Module
 */

/**
 * Example functionality
 */
function task() {
    const net = new Net({
        'architecture': [9, 9, 3],
        'learning_rate': 0.000001
    });
    
    net.addConvolutionalLayer({
        'filter_structure': [3, 3, 1],
        'depth': 3,
        'stride': 1,
        'rectifier': rectifiers.relu,
    });
    
    net.addFullyConnectedLayer({
        'architecture': [2],
        'rectifier': rectifiers.relu,
    });
    
    const targets = {
        'jedi': [1, 0],
        'sith': [0, 1]
    };
    
    const train_promises = [
        net.loadImageDirectory({'directory': './data/training_9/jedi'}),
        net.loadImageDirectory({'directory': './data/training_9/sith'})
    ];
    
    const test_promises = [
        net.loadImageDirectory({'directory': './data/testing_9/sith'}),
        net.loadImageDirectory({'directory': './data/testing_9/jedi'}),
        net.loadImageDirectory({'directory': './data/testing_9/sam'}),
        net.loadImageDirectory({'directory': './data/testing_9/claude'}),
        net.loadImageDirectory({'directory': './data/testing_9/tim'})
    ];
    
    /**
     * Tests and reports accuracy
     * @param {*} sith 
     * @param {*} jedi 
     * @param {*} sam 
     * @param {*} claude 
     * @param {*} tim 
     */
    function test(sith, jedi, sam, claude, tim) {
        sith.forEach((test_image) => {
            const prediction = net.predict(test_image);
            console.log('Test prediction for sith:', prediction);
        });
    
        jedi.forEach((test_image) => {
            const prediction = net.predict(test_image);
            console.log('Test prediction for jedi:', prediction);
        });
    
        claude.forEach((test_image) => {
            const prediction = net.predict(test_image);
            console.log('Test prediction for claude:', prediction);
        });
    
        tim.forEach((test_image) => {
            const prediction = net.predict(test_image);
            console.log('Test prediction for tim:', prediction);
        });
    
        sam.forEach((test_image) => {
            const prediction = net.predict(test_image);
            console.log('Test prediction for sam:', prediction);
        });
    }
    
    let power = 1;
    
    Promise.all(train_promises)
    .then(([jedi_images, sith_images]) => {
        return Promise.all(test_promises)
        .then(([sith, jedi, sam, claude, tim]) => {
            for (var i = 0; i < 500000; i++) {
                if (Math.random() < 0.5) {
                    const jedi_index = Math.floor(Math.random() * jedi_images.length);
                    net.learn(jedi_images[jedi_index], targets.jedi);
                } else {
                    const sith_index = Math.floor(Math.random() * sith_images.length);
                    net.learn(sith_images[sith_index], targets.sith);
                }
    
                if (i % Math.pow(2, power) == 0) {
                    power += 1;
                    test(sith, jedi, sam, claude, tim);
                }
            }
        });
    });
}

task();
