/**
 * ConnectionParams exists so that weights and biases can be shared, as in convolutional layers
 *
 * @class
 * @param {Number} weight - initial weight
 */
class ConnectionParams {
    constructor(weight) {
        this.weight = weight;
    }
}

module.exports = ConnectionParams;
