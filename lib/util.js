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
 * Runs promises while enforcing max concurrency.
 * Does not return results in order.
 * @param {Array<Functino>} fns 
 * @param {Number} maxConcurrency 
 */
async function queueAsyncFunctions(functions, maxConcurrency = 10) {
    let functionsRemaining = [...functions];
    let promisesRunning = [];
    const results = [];
    
    while (functionsRemaining.length || promisesRunning.length) {
        const numPromisesToStart = maxConcurrency - promisesRunning.length;
        const functionsToStart = functionsRemaining.slice(0, numPromisesToStart);
        functionsRemaining = functionsRemaining.slice(numPromisesToStart, numPromisesToStart.length);
        promisesRunning.push(...functionsToStart.map(fn => runPromiseWithContext(fn)));
        await Promise.race(promisesRunning);
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
