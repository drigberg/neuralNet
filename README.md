
# neuralNet

A framework for creating neural networks

## Usage

Create a net:

```javascript
const net = new Net({
    "architecture": [32, 32, 3],
    "learning_rate": 0.02
})
```

Add layers:

```javascript
net.addConvolutionalLayer({
    "filter_structure": [6, 6, 3],
    "depth": 6,
    "stride": 1,
    "rectifier": rectifiers.relu,
})

net.addLayer({
    "num_neurons": 3,
    "rectifier": rectifiers.relu
})
```

Load images and train:

```javascript
net.loadImageDirectory({"directory": "./data/training"})
.then(([jedi, sith]) => {
    for (var i = 0; i < 100; i++) {
        net.learn(jedi[i], [0, 1])
        net.learn(sith[i], [1, 0])
    }
})
```

Implement:

```javascript
net.loadImage("./data/me.png")
.then((image_of_me) => {
    let jedi_or_sith = net.predict(image_of_me)
    console.log(jedi_or_sith)
})
```
