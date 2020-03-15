const {queueAsyncFunctions, runPromisesInSequence} = require('../lib/util');

const { expect } = require('chai');


describe('Utils', () => {
    describe('queueAsyncFunctions', () => {
        it('returns results', async () => {
            const fns = [
                () => new Promise(resolve => resolve(3)),
                () => new Promise(resolve => resolve(4)),
                () => new Promise(resolve => resolve(5)),
                () => new Promise(resolve => resolve(6)),
            ];
            const results = await queueAsyncFunctions(fns);
            expect(results.sort()).to.deep.equal([3, 4, 5, 6]);
        });

        it('returns results when more promises than max concurrency', async () => {
            const fns = [
                () => new Promise(resolve => resolve(3)),
                () => new Promise(resolve => resolve(4)),
                () => new Promise(resolve => resolve(5)),
                () => new Promise(resolve => resolve(6)),
            ];
            const results = await queueAsyncFunctions(fns, 2);
            expect(results.sort()).to.deep.equal([3, 4, 5, 6]);
        });

        it('waits until all are complete', async () => {
            const fns = [
                () => new Promise(resolve => {
                    setTimeout(() => resolve(3), 1);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(4), 5);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(5), 5);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(6), 3);
                }),
            ];
            const results = await queueAsyncFunctions(fns, 2);
            expect(results.sort()).to.deep.equal([3, 4, 5, 6]);
        });

        it('never exceeds max concurrency', async () => {
            let running = 0;
            let maxRunning = 0;

            const fn = () => {
                running += 1;
                if (running > maxRunning) {
                    maxRunning = running;
                }
                return new Promise(resolve => {
                    setTimeout(() => {
                        running -= 1;
                        resolve(3);
                    }, 1);
                });
            };

            const fns = (new Array(100)).fill(fn);
            const results = await queueAsyncFunctions(fns, 10);
            expect(results.sort()).to.deep.equal((new Array(100).fill(3)));
            expect(maxRunning).to.equal(10);
        });

        it('throws error on promise rejection', async () => {
            let error = null;
            const fns = [
                () => new Promise((resolve, reject) => {
                    setTimeout(() => reject(new Error('Error in promise!')), 10);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(4), 5);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(5), 1);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(6), 1);
                }),
            ];
            try {
                await queueAsyncFunctions(fns, 2);
            } catch (err) {
                error = err;
            }
            expect(error).to.not.be.null;
            expect(error.message).to.equal('Error in promise!');
        });
    });

    describe('runPromisesInSequence', () => {
        it('returns results in order', async () => {
            const fns = [
                () => new Promise(resolve => {
                    setTimeout(() => resolve(8), 2);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(2), 2);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(3), 2);
                }),
            ];
            const results = await runPromisesInSequence(fns);
            expect(results).to.deep.equal([8, 2, 3]);
        });

        it('never runs more than one promise at a time', async () => {
            let running = 0;
            let maxRunning = 0;

            const fn = () => {
                running += 1;
                if (running > maxRunning) {
                    maxRunning = running;
                }
                return new Promise(resolve => {
                    setTimeout(() => {
                        running -= 1;
                        resolve(3);
                    }, 1);
                });
            };

            const fns = (new Array(10)).fill(fn);
            const results = await runPromisesInSequence(fns);
            expect(results.sort()).to.deep.equal((new Array(10).fill(3)));
            expect(maxRunning).to.equal(1);
        });

        it('throws error on promise rejection', async () => {
            let error = null;
            const fns = [
                () => new Promise((resolve, reject) => {
                    setTimeout(() => reject(new Error('Error in promise!')), 10);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(4), 5);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(5), 1);
                }),
                () => new Promise(resolve => {
                    setTimeout(() => resolve(6), 1);
                }),
            ];
            try {
                await runPromisesInSequence(fns);
            } catch (err) {
                error = err;
            }
            expect(error).to.not.be.null;
            expect(error.message).to.equal('Error in promise!');
        });
    });
});
