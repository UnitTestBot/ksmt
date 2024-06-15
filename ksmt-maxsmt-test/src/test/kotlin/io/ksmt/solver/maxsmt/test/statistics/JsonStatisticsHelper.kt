package io.ksmt.solver.maxsmt.test.statistics

import com.google.gson.Gson
import java.io.File

internal class JsonStatisticsHelper(private val jsonFile: File) {
    private var firstMet = false
    private var lastMet = false
    private val gson = Gson().newBuilder().setPrettyPrinting().create()

    init {
        if (!jsonFile.exists()) {
            jsonFile.createNewFile()
        }
    }

    fun appendTestStatisticsToFile(statistics: MaxSMTTestStatistics) {
        processBeforeAppending()
        jsonFile.appendText(gson.toJson(statistics))
    }

    fun markLastTestStatisticsAsProcessed() {
        lastMet = true
        jsonFile.appendText("\n]\n}")
    }

    private fun processBeforeAppending() {
        require(!lastMet) { "It's not allowed to append statistics when the last test is processed" }

        if (firstMet) {
            jsonFile.appendText(",")
        } else {
            jsonFile.appendText("{\n\"TESTS\": [\n")
            firstMet = true
        }
    }
}
