/**
 * Run promise with context of completion
 * @param {Function} fn 
 */
function runPromiseWithContext(fn) {
    const promise = fn();
    let complete = false;
    let value = null;
    const result = promise.then((v) => {
        complete = true;
        value = v;
    });
    result.isComplete = () => complete;
    result.getValue = () => value;
    return result;
}

/**
 * Runs promises while enforcing max concurrency
 * Note: does not return results in order.
 * @param {Array<Function>} fns 
 * @param {Number} maxConcurrency 
 */
async function queueAsyncFunctions(functions, maxConcurrency = 10) {
    let functionsRemaining = [...functions];
    let promisesRunning = [];
    const results = [];
    
    while (functionsRemaining.length || promisesRunning.length) {
        // figure out which functions to bump from queue
        const numPromisesToStart = maxConcurrency - promisesRunning.length;
        const functionsToStart = functionsRemaining.slice(0, numPromisesToStart);
        functionsRemaining = functionsRemaining.slice(numPromisesToStart, numPromisesToStart.length);
        promisesRunning.push(...functionsToStart.map(fn => runPromiseWithContext(fn)));

        // wait until first running promise finishes
        await Promise.race(promisesRunning);

        // fetch results and filter running promises 
        const completedPromises = promisesRunning.filter(promise => promise.isComplete());
        results.push(...completedPromises.map(promise => promise.getValue()));
        promisesRunning = promisesRunning.filter(promise => !promise.isComplete());
    }
        
    return results;
}

/**
 * Runs promises and returns results in order
 * @param {Array<Function>} fns - functions which return promises
 */
async function runPromisesInSequence(fns) {
    const results = [];
    for (let i = 0; i < fns.length; i++) {
        results.push(await fns[i]());
    }
    return results;
}

module.exports = {
    queueAsyncFunctions,
    runPromisesInSequence
};

