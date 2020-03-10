// partly based on https://github.com/cazala/synaptic/blob/master/src/Neuron.js

const Net = require('./models/Net');
const rectifiers = require('./lib/rectifiers');

const bias = -0.5;

const net = new Net({
    'input_length': 3,
    'learning_rate': 0.01
});

net.addLayer({
    'bias': bias,
    'num_neurons': 20,
    'rectifier': rectifiers.relu,
});

net.addLayer({
    'bias': bias,
    'num_neurons': 2,
    'rectifier': rectifiers.step,
});

const input = [-100, 50, 3];


net.predict(input)
.then((res) => {
    console.log(res);
});
