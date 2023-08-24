package io.ksmt.solver.neurosmt.runtime.standalone

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.options.required
import com.github.ajalt.clikt.parameters.types.double

class CLArgs : CliktCommand() {
    override fun run() = Unit

    val datasetPath: String by
        option("-D", "--data", help = "path to dataset").required()

    val ordinalsPath: String by
        option("-o", "--ordinals", help = "path to ordinal encoder categories").required()

    val embeddingsPath: String by
        option("-e", "--embeddings", help = "path to embeddings layer").required()

    val convPath: String by
        option("-c", "--conv", help = "path to conv layer").required()

    val decoderPath: String by
        option("-d", "--decoder", help = "path to decoder").required()

    val threshold: Double by
        option("-t", "--threshold", help = "probability threshold for sat/unsat decision")
            .double().default(0.5)
}