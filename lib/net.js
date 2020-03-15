/**
 * Module dependencies
 */

const fs = require('fs');
const PNG = require('pngjs').PNG;

const {ConvolutionalLayer,FullyConnectedLayer,PoolingLayer} = require('./layer');

/**
 * Module
 */

const LAYER_TYPE_MAPPING = {
    'CONVOLUTIONAL': ConvolutionalLayer,
    'POOLING': PoolingLayer,
    'FULLY_CONNECTED': FullyConnectedLayer
};

/**
 * Net class contains layers of neurons
 */
class Net {
    /**
     * 
     * @param {Object} options 
     * @param {Array<Number>} options.architecture
     * @param {Number} options.learningRate
     */
    constructor({ inputArchitecture, layerConfigs, learningRate }) {
        this.learningRate = learningRate;
        this.predictions = 0;
        this.successes = 0;
        this.logPower = 1;

        this.layers = [
            new FullyConnectedLayer({
                isInput: true,
                net: this,
                architecture: inputArchitecture,
            })
        ];

        for (let i = 0; i < layerConfigs.length; i++) {
            const layerConfig = layerConfigs[i];
            const options = {
                net: this,
                inLayer: this.finalLayer,
                ...layerConfig.options
            };

            // const options = layerConfig.options;
            const type = layerConfig.type;
            const validTypes = Object.keys(LAYER_TYPE_MAPPING);
            if (!validTypes.includes(type)) {
                throw new Error(`Invalid layer type ${type}: must be one of ${validTypes.join('', '')}`);
            }
            const LayerClass = LAYER_TYPE_MAPPING[type];
            const layer = new LayerClass(options);
            this.layers.push(layer);
        }
    }

    /**
     * @return {Layer|null}
     */
    get finalLayer() {
        return this.layers.length ? this.layers[this.layers.length - 1] : null;
    }

    /**
     * fetches all layers which are not the input layer
     * @return {Array<Layer>}
     */
    get inputLayer() {
        return this.layers.length ? this.layers[0] : null;
    }

    /**
     * fetches all layers which are not the input layer
     * @return {Array<Layer>}
     */
    get nonInputLayers() {
        return this.layers.slice(1, this.layers.length);
    }

    /**
     * Assigns activations to input layer
     * @param {Array<Number>} input 
     */
    assignInputActivations(input) {
        const inputLayer = this.inputLayer;

        /**
         * Recursive function to assign activations
         * @param {} arch 
         * @param {*} arrOrValue 
         * @param {*} index 
         * @param {*} state 
         */
        function nest(arrOrValue, index, state) {
            index = index || 0;
            if (index < inputLayer.architecture.length) {
                // nest through dimensions
                for (var i = 0; i < inputLayer.architecture[index]; i++) {
                    const nestedState = state ? [state, i].join('.'): String(i);
                    const nestedArr = arrOrValue[i];
                    nest(nestedArr, index + 1, nestedState);
                }
            } else {
                // assign value to neuron
                inputLayer.neuronsByState[state].activation = arrOrValue;
            }
        }

        nest(input);
    }

    /**
     * Generates prediction based on input
     * @param {Array<Number>} input 
     */
    predict(input) {
        if (! (input instanceof Array)) {
            throw new Error('Input must be an array!');
        }

        this.assignInputActivations(input);

        return this.activate();
    }

    /**
     * Activates neurons
     * @return {Array<Number>} returns final prediction
     */
    activate() {
        for (var i = 0; i < this.nonInputLayers.length; i++) {
            this.nonInputLayers[i].activate();
        }

        return this.finalLayer.activate();
    }

    /**
     * Adjusts weights based on error from target
     * @param {Array<Number>} target 
     */
    backPropagate(target) {
        this.finalLayer.propagate(target);

        for (var i = this.nonInputLayers.length - 2; i >= 0; i--) {
            this.nonInputLayers[i].propagate();
        }
    }

    /**
     * Generates prediction and adjusts weights based on error
     * @param {Array<Number>} input 
     * @param {Array<Number>} target 
     */
    learn(input, target) {
        const prediction = this.predict(input);
        let correct = true;
        for (var i = 0; i < target.length; i++) {
            if (target[i] !== prediction[i]) {
                correct = false;
                break;
            }
        }

        this.predictions += 1;
        if (correct) {
            this.successes += 1;
        }

        if (this.predictions % Math.pow(2, this.logPower) == 0) {
            this.logPower += 1;
            console.log(`Prediction #${this.predictions}:`, prediction, '- Target:', target);
            console.log(`Cumulative ${this.successes / this.predictions * 100}% accuracy`);
        }

        this.backPropagate(target);
    }

    /**
     * Loads all images from a directory
     * @param {String} directory 
     */
    loadImageDirectory(directory) {
        const net = this;

        return Promise.all(
            fs.readdirSync(directory)
                .filter((fileName) => fileName[0] !== '.')
                .map((fileName) => net.loadImage(`${directory}/${fileName}`)))
            .catch((err) => {
                console.log('err:', err);
            });
    }

    /**
     * Loads an image
     * @param {String} filePath 
     */
    loadImage(filePath) {
        return new Promise((resolve) => {
            const arr = [];
            fs.createReadStream(filePath)
            .pipe(new PNG({ filterType: 4 }))
            .on('parsed', function() {
                console.log(`Parsed ${filePath}!`);
                for (var x = 0; x < this.width; x++) {
                    arr.push([]);
                    for (var y = 0; y < this.height; y++) {
                        arr[x].push([]);
                        for (var z = 0; z < 3; z++) {
                            const pixelId = ((this.width * y + x) << 2) + z;
                            const pixelValue = this.data[pixelId];
                            // const state = [x, y, z].join('.');
                            arr[x][y].push(pixelValue);
                        }
                    }
                }
                return resolve(arr);
            });
        });
    }
}

/**
 * Module exports
 */

module.exports = {
    Net
};
