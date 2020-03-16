/**
 * Module
 */

const LEVELS = {
    DEBUG: 'DEBUG',
    INFO: 'INFO',
    WARN: 'WARN',
    ERROR: 'ERROR',
    SILENCE: 'SILENCE'
};

const LEVEL_VALUES = {
    DEBUG: 1,
    INFO: 2,
    WARN: 3,
    ERROR: 4,
    SILENCE: 5
};

/**
 * Logger
 * @class
 */
class Logger {
    /**
     * Constructor
     */
    constructor() {
        this.level = LEVELS.DEBUG;
    }

    /**
     * Logs content with label
     * @param {Function} fn - console function
     * @param {String} level - level text to output
     * @param {String} text - content
     */
    _log(fn, level, text) {
        if (LEVEL_VALUES[this.level] <= LEVEL_VALUES[level]) {
            const memoryMB = process.memoryUsage().heapUsed / 1024 / 1024;
            const memoryMBRounded = Math.round(memoryMB * 10) / 10;
            fn(`[${level}] [${memoryMBRounded} MB] ${text}`);
        }
    }

    /**
     * @param {String} level
     */
    setLogLevel(level) {
        if (!Object.values(LEVELS).includes(level)) {
            const levels = Object.keys(LEVELS);
            throw new Error(`Invalid log level! Options are: ${levels.join(', ')}`);
        }
        this.level = level;
    }

    /**
     * Log info
     * @param {String} text 
     */
    info(text) {
        this._log(console.info, 'INFO', text);
    }

    /**
     * Log error
     * @param {String} text 
     */
    error(text) {
        this._log(console.error, 'ERROR', text);
    }

    /**
     * Log warning
     * @param {String} text 
     */
    warn(text) {
        this._log(console.warn, 'WARN', text);
    }

    /**
     * Log debug message
     * @param {String} text 
     */
    debug(text) {
        this._log(console.debug, 'DEBUG', text);
    }
}

/**
 * Module exports
 */

module.exports = {
    logger: new Logger(),
    LEVELS
};
