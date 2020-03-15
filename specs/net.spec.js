
const fs = require('fs');
const path = require('path');
const { expect } = require('chai');

const {Net} = require('../lib/net');

describe('Net', () => {
    describe('loadImage', () => {
        it('loads image', async () => {
            const net = new Net({
                inputArchitecture: [9, 9, 3],
                learningRate: 0,
                layerConfigs: []
            });
            const data = await net.loadImage(path.join(__dirname, '/data/images/9x9x3.png'));
            const expectedImageData = JSON.parse(fs.readFileSync(path.join(__dirname, '/data/9x9x3.json')));
            expect(data).to.deep.equal(expectedImageData);
        });
    });
    describe('loadImageDirectory', () => {
        it('loads all images in directory', async () => {
            const net = new Net({
                inputArchitecture: [9, 9, 3],
                learningRate: 0,
                layerConfigs: []
            });
            const data = await net.loadImageDirectory(path.join(__dirname, '/data/images'));
            const expectedImageData = JSON.parse(fs.readFileSync(path.join(__dirname, '/data/9x9x3.json')));
            expect(data).to.deep.equal([expectedImageData]);
        });
    });
    describe('loadImageDirectories', () => {
        it('loads all images in multiple directories', async () => {
            const net = new Net({
                inputArchitecture: [9, 9, 3],
                learningRate: 0,
                layerConfigs: []
            });
            const data = await net.loadImageDirectories([path.join(__dirname, '/data/images')]);
            const expectedImageData = JSON.parse(fs.readFileSync(path.join(__dirname, '/data/9x9x3.json')));
            expect(data).to.deep.equal([[expectedImageData]]);
        });
    });
});
