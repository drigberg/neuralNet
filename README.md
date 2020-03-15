
# neuralNet

A framework for creating neural networks

Cool things:
- Only requires pngjs (+ chai and eslint for development)
- Can create fully-connected and convolutional layers
- Can accept input matrices with any number of dimensions

## Usage

Create a net:

```javascript
const net = new Net({
    architecture: [32, 32, 3],
    learningRate: 0.00001,
    layerConfigs: [
        {
            type: 'CONVOLUTIONAL',
            options: {
                filterArchitecture: [3, 3, 1],
                depth: 2,
                stride: 1,
                rectifier: rectifiers.relu,
            }
        },
        {
            type: 'FULLY_CONNECTED',
            options: {
                architecture: [2],
                rectifier: rectifiers.step,
            }
        }
    ]
})
```


Load images and train:

```javascript
let trainPromises = [
    net.loadImageDirectory({"directory": "./data/training_9/jedi"}),
    net.loadImageDirectory({"directory": "./data/training_9/sith"})
]

Promise.all(trainPromises)
.then(([jediImages, sithImages]) => {
    for (var i = 0; i < 100; i++) {
        net.learn(jediImages[i], [0, 1])
        net.learn(sithImages[i], [1, 0])
    }
})
```

Implement:

```javascript
net.loadImage("./data/me.png")
.then((imageOfMe) => {
    let jediOrSith = net.predict(imageOfMe)
    console.log(jediOrSith)
})
```

## To-Do List

- Better name
    - dandrites-js
- MSE
- Pooling layers
- Export && import weights
- Export prediction function
- Visualisation