/**
 * Module
 */

const LEVELS = {
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
        fn(`[${level}] ${text}`);
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
        if (this.level <= LEVELS.INFO) {
            this._log(console.info, 'INFO', text);
        }
    }

    /**
     * Log error
     * @param {String} text 
     */
    error(text) {
        if (this.level <= LEVELS.ERROR) {
            this._log(console.error, 'ERROR', text);
        }
    }

    /**
     * Log warning
     * @param {String} text 
     */
    warn(text) {
        if (this.level <= LEVELS.WARN) {
            this._log(console.warn, 'WARN', text);
        }
    }

    /**
     * Log debug message
     * @param {String} text 
     */
    debug(text) {
        if (this.level <= LEVELS.DEBUG) {
            this._log(console.debug, 'DEBUG', text);
        }
    }
}

/**
 * Module exports
 */

module.exports = {
    logger: new Logger(),
    LEVELS
};
