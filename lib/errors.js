/**
 * Module
 */

const codes = {
    INCOMPATIBLE_FILTER: 101,
};

const errors = {
    INCOMPATIBLE_FILTER: ({ inputLength, filterLength, stride, res }) => {
        const explanation = `(${inputLength} - (${filterLength} - ${stride})) / ${stride} = ${res}`;

        const err = new Error(`Incompatible combination of input structure, filter structure, and stride: ${explanation}`);
        err.code = codes.INCOMPATIBLE_FILTER;

        return err;
    },
};

/**
 * Module exports
 */

module.exports = {
    errors,
    codes
};