
# neuralNet

A framework for creating neural networks

## Usage

Create a net:

```javascript
const net = new Net({
    "architecture": [32, 32, 3],
    "learning_rate": 0.00001
})
```

Add layers:

```javascript
net.addConvolutionalLayer({
    "filter_structure": [3, 3, 1],
    "depth": 2,
    "stride": 1,
    "rectifier": rectifiers.relu,
})

net.addFullyConnectedLayer({
    "architecture": [2],
    "rectifier": rectifiers.step,
})
```

Load images and train:

```javascript
let train_promises = [
    net.loadImageDirectory({"directory": "./data/training_9/jedi"}),
    net.loadImageDirectory({"directory": "./data/training_9/sith"})
]

Promise.all(train_promises)
.then(([jedi_images, sith_images]) => {
    for (var i = 0; i < 100; i++) {
        net.learn(jedi_images[i], [0, 1])
        net.learn(sith_images[i], [1, 0])
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

## To-Do List

- Better name
	- deep-node
	- net-js
    - dandrites-js
- MSE
- Conv net
	- Multi-dimensional inputs
	- Weight object used by connections
		- allows for shared weights
	- Pooling layers
	- Export weights
	- Import weights
	- Visualisation
	 	- Instance method, or static, using exported weights?
