/**
 * Module
 */

const codes = {
    'INCOMPATIBLE_FILTER': 101,
};

const errors = {
    'INCOMPATIBLE_FILTER': ({ input_length, filter_length, stride, res }) => {
        const explanation = `(${input_length} - (${filter_length} - ${stride})) / ${stride} = ${res}`;

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