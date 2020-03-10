/**
 * Module exports
 */

module.exports = {
    'sigmoid': function (x, derive) {
        var fx = 1 / (1 + Math.exp(-x));
        if (!derive)
            {return fx;}
        return fx * (1 - fx);
    },
    'tanh': function (x, derive) {
        if (derive)
            {return 1 - Math.pow(Math.tanh(x), 2);}
        return Math.tanh(x);
    },
    'identity': function (x, derive) {
        return derive ? 1 : x;
    },
    'step': function (x, derive) {
        return derive ? 1 : x > 0 ? 1 : 0;
    },
    'relu': function (x, derive) {
        if (derive)
            {return x > 0 ? 1 : 0;}
        return x > 0 ? x : 0;
    }
};
