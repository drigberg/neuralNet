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
 * Tests and reports accuracy
 * @param {Net} net
 * @param {*} sith 
 * @param {*} jedi 
 * @param {*} sam 
 * @param {*} claude 
 * @param {*} tim 
 */
function test(net, sith, jedi, sam, claude, tim) {
    sith.forEach((testImage) => {
        const prediction = net.predict(testImage);
        logger.info('Test prediction for sith:', prediction);
    });

    jedi.forEach((testImage) => {
        const prediction = net.predict(testImage);
        logger.info('Test prediction for jedi:', prediction);
    });

    claude.forEach((testImage) => {
        const prediction = net.predict(testImage);
        logger.info('Test prediction for claude:', prediction);
    });

    tim.forEach((testImage) => {
        const prediction = net.predict(testImage);
        logger.info('Test prediction for tim:', prediction);
    });

    sam.forEach((testImage) => {
        const prediction = net.predict(testImage);
        logger.info('Test prediction for sam:', prediction);
    });
}

/**
 * Example functionality
 */
async function task() {
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
        jedi: [1, 0],
        sith: [0, 1]
    };
    
    let power = 1;

    const [jediTrainingImages, sithTrainingImages] = await net.loadImageDirectories([
        './data/starwars_32x32/training/jedi',
        './data/starwars_32x32/training/sith'
    ]);

    const [
        sithTestingImages,
        jediTestingImages,
        samTestingImages,
        claudeTestingImages,
        timTestingImages] = await net.loadImageDirectories([
            './data/starwars_32x32/testing/sith',
            './data/starwars_32x32/testing/jedi',
            './data/starwars_32x32/testing/sam',
            './data/starwars_32x32/testing/claude',
            './data/starwars_32x32/testing/tim'
        ]);

    for (let i = 0; i < 100000; i++) {
        if (Math.random() < 0.5) {
            const jediIndex = Math.floor(Math.random() * jediTrainingImages.length);
            net.learn(jediTrainingImages[jediIndex], targets.jedi);
        } else {
            const sithIndex = Math.floor(Math.random() * sithTrainingImages.length);
            net.learn(sithTrainingImages[sithIndex], targets.sith);
        }

        if (i % Math.pow(2, power) == 0) {
            power += 1;
            test(net, sithTestingImages, jediTestingImages, samTestingImages, claudeTestingImages, timTestingImages);
        }
    }
}

task();
