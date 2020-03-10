/**
 * Module dependencies
 */

const fs = require('fs');
const PNG = require('pngjs').PNG;

const {ConvolutionalLayer,FullyConnectedLayer} = require('./Layer');

/**
 * Module
 */


class Net {
    constructor({ architecture, learning_rate }) {
        const input_args = {
            'is_input': true,
            'architecture': architecture,
            'net': this
        };

        this.predictions = 0;
        this.successes = 0;
        this.logPower = 1;

        this.input_layer = new FullyConnectedLayer(input_args);
        this.layers = [];
        this.learning_rate = learning_rate;
    }

    get finalLayer() {
        return this.layers.length ? this.layers[this.layers.length - 1] : null;
    }

    navigateInput(input) {
        const input_layer = this.input_layer;
        nest(input_layer.architecture, input);

        function nest(arch, arr_or_value, index, state) {
            index = index || 0;
            if (index < arch.length) {
                for (var i = 0; i < arch[index]; i++) {
                    const nested_state = state ? [state, i].join('.'): String(i);
                    const nested_arr = arr_or_value[i];
                    nest(arch, nested_arr, index + 1, nested_state);
                }
            } else {
                input_layer.neurons[state].activation = arr_or_value;
                
            }
        }
    }

    predict(input) {
        if (! (input instanceof Array)) {
            throw new Error('Input must be an array!');
        }

        this.navigateInput(input);

        return this.activate();
    }

    activate() {
        for (var i = 0; i < this.layers.length; i++) {
            this.layers[i].activate();
        }

        return this.finalLayer.activate();
    }

    backPropagate(target) {
        this.finalLayer.propagate(target);

        for (var i = this.layers.length - 2; i >= 0; i--) {
            this.layers[i].propagate();
        }
    }

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
        this.successes += correct ? 1 : 0;

        if (this.predictions % Math.pow(2, this.logPower) == 0) {
            this.logPower += 1;
            console.log(`Prediction #${this.predictions}:`, prediction, '- Target:', target);
            console.log(`Cumulative ${this.successes / this.predictions * 100}% accuracy`);
        }

        this.backPropagate(target);
    }

    addFullyConnectedLayer(layer_args) {
        layer_args = this.supplementLayerArgs(layer_args);

        this.layers.push(new FullyConnectedLayer(layer_args));
    }

    addConvolutionalLayer(layer_args) {
        layer_args = this.supplementLayerArgs(layer_args);

        this.layers.push(new ConvolutionalLayer(layer_args));
    }

    supplementLayerArgs(layer_args) {
        const in_layer = this.finalLayer || this.input_layer;

        Object.assign(layer_args, {
            'in_layer': in_layer,
            'net': this
        });

        return layer_args;
    }

    loadImageDirectory({ directory }) {
        const net = this;

        return Promise.all(
            fs.readdirSync(directory)
                .filter((file_name) => file_name[0] !== '.')
                .map((file_name) => net.loadImage(`${directory}/${file_name}`)))
            .catch((err) => {
                console.log('err:', err);
            });
    }

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
