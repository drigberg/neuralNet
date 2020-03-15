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
        inputArchitecture: [9, 9, 3],
        learningRate: 0.000001,
        layerConfigs: [
            {
                type: 'CONVOLUTIONAL',
                options: {
                    filterArchitecture: [3, 3, 1],
                    depth: 3,
                    stride: 1,
                    rectifier: rectifiers.relu,
                }
            },
            {
                type: 'FULLY_CONNECTED',
                options: {
                    architecture: [2],
                    rectifier: rectifiers.relu,
                }
            }
        ]
    });
    
    const targets = {
        'jedi': [1, 0],
        'sith': [0, 1]
    };
    
    const trainPromises = [
        net.loadImageDirectory({'directory': './data/training_9/jedi'}),
        net.loadImageDirectory({'directory': './data/training_9/sith'})
    ];
    
    const testPromises = [
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
        sith.forEach((testImage) => {
            const prediction = net.predict(testImage);
            console.log('Test prediction for sith:', prediction);
        });
    
        jedi.forEach((testImage) => {
            const prediction = net.predict(testImage);
            console.log('Test prediction for jedi:', prediction);
        });
    
        claude.forEach((testImage) => {
            const prediction = net.predict(testImage);
            console.log('Test prediction for claude:', prediction);
        });
    
        tim.forEach((testImage) => {
            const prediction = net.predict(testImage);
            console.log('Test prediction for tim:', prediction);
        });
    
        sam.forEach((testImage) => {
            const prediction = net.predict(testImage);
            console.log('Test prediction for sam:', prediction);
        });
    }
    
    let power = 1;
    
    Promise.all(trainPromises)
    .then(([jediImages, sithImages]) => {
        return Promise.all(testPromises)
        .then(([sith, jedi, sam, claude, tim]) => {
            for (let i = 0; i < 500000; i++) {
                if (Math.random() < 0.5) {
                    const jediIndex = Math.floor(Math.random() * jediImages.length);
                    net.learn(jediImages[jediIndex], targets.jedi);
                } else {
                    const sithIndex = Math.floor(Math.random() * sithImages.length);
                    net.learn(sithImages[sithIndex], targets.sith);
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
