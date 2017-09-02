
# neuralNet

A framework for creating neural networks

## Usage

Create a net:

```javascript
let net = new Net({
    "input_length": 2,
    "learning_rate": 0.05
})
```

Add layers:

```javascript
net.addLayer({
    "num_neurons": 3,
    "rectifier": rectifiers.relu
})

net.addLayer({
    "num_neurons": 1,
    "rectifier": rectifiers.step,
})
```

Plug in input and target arrays for training:

```javascript
net.learn(input, target)
```

Implement:

```javascript
net.predict(input)
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
