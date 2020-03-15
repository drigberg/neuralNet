
# neuralNet

A framework for creating neural networks

Cool things:
- Only requires pngjs in production
- Can create fully-connected, convolutional, and pooling layers
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
net.loadImageDirectories(["./data/training_9/jedi", "./data/training_9/sith"])
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
    const result = net.predict(imageOfMe)
    console.log(`Jedi or sith: ${result}`)
})
```

## To-Do List

- Better name
    - dandrites-js
- MSE
- Export && import weights
- Export prediction function
- Visualisation