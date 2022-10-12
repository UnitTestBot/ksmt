package org.ksmt.runner.core

import com.jetbrains.rd.util.ConsoleLoggerFactory
import com.jetbrains.rd.util.ILoggerFactory
import com.jetbrains.rd.util.LogLevel
import com.jetbrains.rd.util.Logger
import com.jetbrains.rd.util.Statics

class LoggerFactory private constructor(
    private val factory: ILoggerFactory,
    private var minLevel: LogLevel
) : ILoggerFactory {

    val level: LogLevel
        get() = minLevel

    override fun getLogger(category: String): Logger = object : Logger {
        private val realLog = factory.getLogger(category)

        private val minEnabledLevel by lazy {
            LogLevel.values().firstOrNull { realLog.isEnabled(it) } ?: LogLevel.Fatal
        }

        override fun isEnabled(level: LogLevel): Boolean = level >= minLevel

        private fun ensureLogLevelEnabled(level: LogLevel) =
            if (level < minEnabledLevel) minEnabledLevel else level

        override fun log(level: LogLevel, message: Any?, throwable: Throwable?) {
            realLog.log(ensureLogLevelEnabled(level), message, throwable)
        }
    }

    fun <T> withLoglevel(level: LogLevel, block: () -> T): T {
        val prevLevel = minLevel
        val prevConsoleLogLevel = ConsoleLoggerFactory.minLevelToLog
        return try {
            ConsoleLoggerFactory.minLevelToLog = level
            minLevel = level
            block()
        } finally {
            ConsoleLoggerFactory.minLevelToLog = prevConsoleLogLevel
            minLevel = prevLevel
        }
    }

    companion object {
        private fun leveledFactory(minLevel: LogLevel): LoggerFactory {
            ConsoleLoggerFactory.minLevelToLog = minLevel
            val factory = Statics<ILoggerFactory>().get() ?: ConsoleLoggerFactory
            return LoggerFactory(factory, minLevel)
        }

        fun create(minLevel: LogLevel): LoggerFactory {
            val factory = leveledFactory(minLevel)
            Statics<ILoggerFactory>().push(factory)
            return factory
        }
    }
}
