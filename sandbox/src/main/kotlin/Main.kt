import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import com.jetbrains.rd.framework.SerializationCtx
import com.jetbrains.rd.framework.Serializers
import com.jetbrains.rd.framework.SocketWire.Companion.timeout
import com.jetbrains.rd.framework.UnsafeBuffer
import io.ksmt.KContext
import io.ksmt.expr.*
import io.ksmt.runner.serializer.AstSerializationCtx
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.neurosmt.getAnswerForTest
import io.ksmt.solver.neurosmt.runtime.NeuroSMTModelRunner
import io.ksmt.solver.z3.*
import io.ksmt.sort.*
import io.ksmt.utils.getValue
import io.ksmt.utils.uncheckedCast
import me.tongfei.progressbar.ProgressBar
import java.io.*
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicLong
import kotlin.io.path.isRegularFile
import kotlin.io.path.name
import kotlin.io.path.pathString
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

fun serialize(ctx: KContext, expressions: List<KExpr<KBoolSort>>, outputStream: OutputStream) {
    val serializationCtx = AstSerializationCtx().apply { initCtx(ctx) }
    val marshaller = AstSerializationCtx.marshaller(serializationCtx)
    val emptyRdSerializationCtx = SerializationCtx(Serializers())

    val buffer = UnsafeBuffer(ByteArray(20_000)) // ???

    expressions.forEach { expr ->
        marshaller.write(emptyRdSerializationCtx, buffer, expr)
    }

    outputStream.write(buffer.getArray())
    outputStream.flush()
}

fun deserialize(ctx: KContext, inputStream: InputStream): List<KExpr<KBoolSort>> {
    val srcSerializationCtx = AstSerializationCtx().apply { initCtx(ctx) }
    val srcMarshaller = AstSerializationCtx.marshaller(srcSerializationCtx)
    val emptyRdSerializationCtx = SerializationCtx(Serializers())

    val buffer = UnsafeBuffer(inputStream.readBytes())
    val expressions: MutableList<KExpr<KBoolSort>> = mutableListOf()

    while (true) {
        try {
            expressions.add(srcMarshaller.read(emptyRdSerializationCtx, buffer).uncheckedCast())
        } catch (e : IllegalStateException) {
            break
        }
    }

    return expressions
}

object MethodNameStorage {
    val methodName = ThreadLocal<String>()
}

class LogSolver<C : KSolverConfiguration>(
    private val ctx: KContext, private val baseSolver: KSolver<C>
) : KSolver<C> by baseSolver {

    companion object {
        val counter = ThreadLocal<Long>()
    }

    init {
        File("formulas").mkdirs()
        File("formulas/${MethodNameStorage.methodName.get()}").mkdirs()

        counter.set(0)
    }

    private fun getNewFileCounter(): Long {
        val result = counter.get()
        counter.set(result + 1)
        return result
    }

    private fun getNewFileName(suffix: String): String {
        return "formulas/${MethodNameStorage.methodName.get()}/f-${getNewFileCounter()}-$suffix"
    }

    val stack = mutableListOf<MutableList<KExpr<KBoolSort>>>(mutableListOf())

    override fun assert(expr: KExpr<KBoolSort>) {
        stack.last().add(expr)
        baseSolver.assert(expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        stack.last().add(expr)
        baseSolver.assertAndTrack(expr)
    }

    override fun push() {
        stack.add(mutableListOf())
        baseSolver.push()
    }

    override fun pop(n: UInt) {
        repeat(n.toInt()) {
            stack.removeLast()
        }
        baseSolver.pop(n)
    }

    override fun check(timeout: Duration): KSolverStatus {
        val result = baseSolver.check(timeout)
        serialize(ctx, stack.flatten(), FileOutputStream(getNewFileName(result.toString().lowercase())))
        return result
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        val result = baseSolver.checkWithAssumptions(assumptions, timeout)
        serialize(ctx, stack.flatten() + assumptions, FileOutputStream(getNewFileName(result.toString().lowercase())))
        return result
    }
}

const val THRESHOLD = 0.5

fun main() {

    val ctx = KContext(
        astManagementMode = KContext.AstManagementMode.NO_GC,
        simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY
    )

    val pathToDataset = "formulas"
    val files = Files.walk(Path.of(pathToDataset)).filter { it.isRegularFile() }.toList()

    val runner = NeuroSMTModelRunner(
        ctx,
        ordinalsPath = "usvm-enc-2.cats",
        embeddingPath = "embeddings.onnx",
        convPath = "conv.onnx",
        decoderPath = "decoder.onnx"
    )

    var sat = 0; var unsat = 0; var skipped = 0
    var ok = 0; var wa = 0

    val confusionMatrix = mutableMapOf<Pair<KSolverStatus, KSolverStatus>, Int>()

    files.forEachIndexed sample@{ ind, it ->
        if (ind % 100 == 0) {
            println("#$ind: $ok / $wa [$sat / $unsat / $skipped]")
            println(confusionMatrix)
            println()
        }

        val sampleFile = File(it.pathString)

        val assertList = try {
            deserialize(ctx, FileInputStream(sampleFile))
        } catch (e: Exception) {
            skipped++
            return@sample
        }

        val answer = when {
            it.name.endsWith("-sat") -> KSolverStatus.SAT
            it.name.endsWith("-unsat") -> KSolverStatus.UNSAT
            else -> KSolverStatus.UNKNOWN
        }

        if (answer == KSolverStatus.UNKNOWN) {
            skipped++
            return@sample
        }

        val prob = with(ctx) {
            val formula = when (assertList.size) {
                0 -> {
                    skipped++
                    return@sample
                }
                1 -> {
                    assertList[0]
                }
                else -> {
                    mkAnd(assertList)
                }
            }

            runner.run(formula)
        }

        val output = if (prob < THRESHOLD) {
            KSolverStatus.UNSAT
        } else {
            KSolverStatus.SAT
        }

        when (answer) {
            KSolverStatus.SAT -> sat++
            KSolverStatus.UNSAT -> unsat++
            else -> { /* can't happen */ }
        }

        if (output == answer) {
            ok++
        } else {
            wa++
        }

        confusionMatrix.compute(answer to output) { _, v ->
            if (v == null) {
                1
            } else {
                v + 1
            }
        }
    }

    println()
    println("sat: $sat; unsat: $unsat; skipped: $skipped")
    println("ok: $ok; wa: $wa")

    return

    /*
    with(ctx) {
        val a by boolSort
        val b by intSort
        val c by intSort
        val d by realSort
        val e by bv8Sort
        val f by fp64Sort
        val g by mkArraySort(realSort, boolSort)
        val h by fp64Sort

        val expr = mkBvXorExpr(mkBvShiftLeftExpr(e, mkBv(1.toByte())), mkBvNotExpr(e)) eq
                mkBvLogicalShiftRightExpr(e, mkBv(1.toByte()))

        val runner = NeuroSMTModelRunner(ctx, "usvm-enc-2.cats", "embeddings.onnx", "conv.onnx", "decoder.onnx")
        println(runner.run(expr))
    }

    return

     */

    val env = OrtEnvironment.getEnvironment()
    //val session = env.createSession("kek.onnx")
    val session = env.createSession("conv.onnx")

    println(session.inputNames)
    for (info in session.inputInfo) {
        println(info)
    }

    println()

    println(session.outputNames)
    for (info in session.outputInfo) {
        println(info)
    }

    println()

    println(session.metadata)
    println()

    //val nodeLabels = listOf(listOf(0L), listOf(1L), listOf(2L), listOf(3L), listOf(4L), listOf(5L), listOf(6L))
    val nodeLabels = listOf(listOf(0L), listOf(1L), listOf(2L))
    //val nodeFeatures = (1..7).map { (0..31).map { it / 31.toFloat() } }
    val nodeFeatures = (1..3).map { (0..31).map { it / 31.toFloat() } }
    val edges = listOf(
        listOf(0L, 1L),
        listOf(2L, 2L)
        //listOf(0L, 1L, 2L, 3L, 4L, 5L),
        //listOf(1L, 2L, 3L, 4L, 5L, 6L)
    )
    val depths = listOf(1L)
    val rootPtrs = listOf(0L, 3L)

    /*
    val nodeLabels = listOf(listOf(0L), listOf(1L), listOf(1L), listOf(0L))
    val edges = listOf(
        listOf(0L, 0L),
        listOf(1L, 1L)
    )
    val depths = listOf(1L, 1L)
    val rootPtrs = listOf(0L, 1L, 2L)
    */

    val nodeLabelsBuffer = LongBuffer.allocate(nodeLabels.sumOf { it.size })
    nodeLabels.forEach { features ->
        features.forEach { feature ->
            nodeLabelsBuffer.put(feature)
        }
    }
    nodeLabelsBuffer.rewind()

    val nodeFeaturesBuffer = FloatBuffer.allocate(nodeFeatures.sumOf { it.size })
    nodeFeatures.forEach { features ->
        features.forEach { feature ->
            nodeFeaturesBuffer.put(feature)
        }
    }
    nodeFeaturesBuffer.rewind()

    val edgesBuffer = LongBuffer.allocate(edges.sumOf { it.size })
    edges.forEach { row ->
        row.forEach { node ->
            edgesBuffer.put(node)
        }
    }
    edgesBuffer.rewind()

    val depthsBuffer = LongBuffer.allocate(depths.size)
    depths.forEach { d ->
        depthsBuffer.put(d)
    }
    depthsBuffer.rewind()

    val rootPtrsBuffer = LongBuffer.allocate(rootPtrs.size)
    rootPtrs.forEach { r ->
        rootPtrsBuffer.put(r)
    }
    rootPtrsBuffer.rewind()

    val nodeLabelsData = OnnxTensor.createTensor(env, nodeLabelsBuffer, listOf(nodeLabels.size.toLong(), nodeLabels[0].size.toLong()).toLongArray())
    val nodeFeaturesData = OnnxTensor.createTensor(env, nodeFeaturesBuffer, listOf(nodeFeatures.size.toLong(), nodeFeatures[0].size.toLong()).toLongArray())
    val edgesData = OnnxTensor.createTensor(env, edgesBuffer, listOf(edges.size.toLong(), edges[0].size.toLong()).toLongArray())
    val depthsData = OnnxTensor.createTensor(env, depthsBuffer, listOf(depths.size.toLong()).toLongArray())
    val rootPtrsData = OnnxTensor.createTensor(env, rootPtrsBuffer, listOf(rootPtrs.size.toLong()).toLongArray())

    /*
    val result = session.run(mapOf("node_labels" to nodeLabelsData, "edges" to edgesData, "depths" to depthsData, "root_ptrs" to rootPtrsData))
    val output = (result.get("output").get().value as Array<*>).map {
        (it as FloatArray).toList()
    }
    */

    var curFeatures = nodeFeaturesData
    repeat(10) {
        val result = session.run(mapOf("node_features" to curFeatures, "edges" to edgesData))
        curFeatures = result.get(0) as OnnxTensor
        println(curFeatures.info.shape.toList())
        curFeatures.info.shape
        println(curFeatures.floatBuffer.array().toList().subList(64, 96))
    }

    /*
    val output = (result.get("out").get().value as Array<*>).map {
        (it as FloatArray).toList()
    }

    println(output)
    */


    /*
    val ctx = KContext()

    with(ctx) {
        val files = Files.walk(Path.of("/home/stephen/Desktop/formulas")).filter { it.isRegularFile() }

        var ok = 0; var fail = 0
        files.forEach {
            try {
                println(deserialize(ctx, FileInputStream(it.toFile())).size)
                ok++
            } catch (e: Exception) {
                fail++
            }
        }
        println("$ok / $fail")
    }*/

    /*
    with(ctx) {
        // create symbolic variables
        val a by boolSort
        val b by intSort
        val c by intSort
        val d by realSort
        val e by bv8Sort
        val f by fp64Sort
        val g by mkArraySort(realSort, boolSort)
        val h by fp64Sort

        val x by intSort

        // create an expression
        val constraint = a and (b ge c + 3.expr) and (b * 2.expr gt 8.expr) and
                ((e eq mkBv(3.toByte())) or (d neq mkRealNum("271/100"))) and
                (f eq 2.48.expr) and
                (d + b.toRealExpr() neq mkRealNum("12/10")) and
                (mkArraySelect(g, d) eq a) and
                (mkArraySelect(g, d + mkRealNum(1)) neq a) and
                (mkArraySelect(g, d - mkRealNum(1)) eq a) and
                (d * d eq mkRealNum(4)) and
                (mkBvMulExpr(e, e) eq mkBv(9.toByte()))

        val formula = """
            (declare-fun x () Real)
            (declare-fun y () Real)
            (declare-fun z () Real)
            (declare-fun a () Int)
            (assert (or (and (= y (+ x z)) (= x (+ y z))) (= 2.71 x)))
            (assert (= a 2))
            (check-sat)
        """

        val assertions = mkAnd(KZ3SMTLibParser(this).parse(formula))

        val bvExpr = mkBvXorExpr(mkBvShiftLeftExpr(e, mkBv(1.toByte())), mkBvNotExpr(e)) eq
                mkBvLogicalShiftRightExpr(e, mkBv(1.toByte()))

        val buf = ByteArrayOutputStream()
        serialize(ctx, listOf(constraint, assertions, bvExpr), buf)
        deserialize(ctx, ByteArrayInputStream(buf.toByteArray())).forEach {
            println("nxt: $it")
        }

        /*
        KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(mkAnd(assertions)).forEach {
            println("${it.name} | ${it.argSorts} | ${it.sort}")
        }
        */

        KZ3Solver(this).use { solver ->
            LogSolver(this, solver).use { solver ->
                solver.assert(constraint)

                val satisfiability = solver.check(timeout = 180.seconds)
                println(satisfiability)

                if (satisfiability == KSolverStatus.SAT) {
                    val model = solver.model()
                    println(model)
                }
            }
        }

        /*
        KNeuroSMTSolver(this).use { solver -> // create a Stub SMT solver instance
            // assert expression
            solver.assert(constraint)

            // check assertions satisfiability with timeout
            val satisfiability = solver.check(timeout = 1.seconds)
            // println(satisfiability) // SAT
        }
        */
    }
    */
}