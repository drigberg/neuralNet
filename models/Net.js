/**
 * Module dependencies
 */

const fs = require('fs');
const PNG = require('pngjs').PNG;

const {ConvolutionalLayer,FullyConnectedLayer,PoolingLayer} = require('./Layer');

/**
 * Module
 */


/**
 * Net class contains layers of neurons
 */
class Net {
    /**
     * 
     * @param {Object} options 
     * @param {Array<Number>} options.architecture
     * @param {Number} options.learning_rate
     */
    constructor({ architecture, learning_rate }) {
        const input_args = {
            is_input: true,
            net: this,
            architecture,
        };

        this.predictions = 0;
        this.successes = 0;
        this.logPower = 1;

        this.input_layer = new FullyConnectedLayer(input_args);
        this.layers = [];
        this.learning_rate = learning_rate;
    }

    /**
     * @return {Layer|null}
     */
    get finalLayer() {
        return this.layers.length ? this.layers[this.layers.length - 1] : null;
    }

    /**
     * Assigns activations to input layer
     * @param {Array<Number>} input 
     */
    assignInputActivations(input) {
        const input_layer = this.input_layer;

        /**
         * Recursive function to assign activations
         * @param {} arch 
         * @param {*} arr_or_value 
         * @param {*} index 
         * @param {*} state 
         */
        function nest(arr_or_value, index, state) {
            index = index || 0;
            if (index < input_layer.architecture.length) {
                // nest through dimensions
                for (var i = 0; i < input_layer.architecture[index]; i++) {
                    const nested_state = state ? [state, i].join('.'): String(i);
                    const nested_arr = arr_or_value[i];
                    nest(nested_arr, index + 1, nested_state);
                }
            } else {
                // assign value to neuron
                input_layer.neurons[state].activation = arr_or_value;
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
        for (var i = 0; i < this.layers.length; i++) {
            this.layers[i].activate();
        }

        return this.finalLayer.activate();
    }

    /**
     * Adjusts weights based on error from target
     * @param {Array<Number>} target 
     */
    backPropagate(target) {
        this.finalLayer.propagate(target);

        for (var i = this.layers.length - 2; i >= 0; i--) {
            this.layers[i].propagate();
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
     * Adds a fully connected layer
     * @param {Object} layer_args 
     */
    addFullyConnectedLayer(layer_args) {
        const supplemented_args = this.getSupplementedLayerArgs(layer_args);
        this.layers.push(new FullyConnectedLayer({
            ...layer_args,
            ...supplemented_args
        }));
    }

    /**
     * Adds a convolutional layer
     * @param {Object} layer_args 
     */
    addConvolutionalLayer(layer_args) {
        const supplemented_args = this.getSupplementedLayerArgs(layer_args);
        this.layers.push(new ConvolutionalLayer({
            ...layer_args,
            ...supplemented_args
        }));
    }

    /**
     * Adds a pooling layer
     * @param {Object} layer_args 
     */
    addPoolingLayer(layer_args) {
        const supplemented_args = this.getSupplementedLayerArgs(layer_args);
        this.layers.push(new PoolingLayer({
            ...layer_args,
            ...supplemented_args
        }));
    }

    /**
     * Returns arguments needed for layers
     * @return {Object}
     */
    getSupplementedLayerArgs() {
        return {
            net: this,
            in_layer: this.finalLayer || this.input_layer
        };
    }

    /**
     * Loads all images from a directory
     * @param {String} directory 
     */
    loadImageDirectory(directory) {
        const net = this;

        return Promise.all(
            fs.readdirSync(directory)
                .filter((file_name) => file_name[0] !== '.')
                .map((file_name) => net.loadImage(`${directory}/${file_name}`)))
            .catch((err) => {
                console.log('err:', err);
            });
    }

    /**
     * Loads an image
     * @param {String} file_path 
     */
    loadImage(file_path) {
        return new Promise((resolve) => {
            const arr = [];
            fs.createReadStream(file_path)
            .pipe(new PNG({ filterType: 4 }))
            .on('parsed', function() {
                console.log(`Parsed ${file_path}!`);
                for (var x = 0; x < this.width; x++) {
                    arr.push([]);
                    for (var y = 0; y < this.height; y++) {
                        arr[x].push([]);
                        for (var z = 0; z < 3; z++) {
                            const pixel_id = ((this.width * y + x) << 2) + z;
                            const pixel_value = this.data[pixel_id];
                            // const state = [x, y, z].join('.');
                            arr[x][y].push(pixel_value);
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
