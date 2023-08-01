import com.jetbrains.rd.framework.SerializationCtx
import com.jetbrains.rd.framework.Serializers
import com.jetbrains.rd.framework.UnsafeBuffer
import io.ksmt.KContext
import io.ksmt.expr.*
import io.ksmt.runner.serializer.AstSerializationCtx
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.neurosmt.KNeuroSMTSolver
import io.ksmt.solver.z3.*
import io.ksmt.sort.*
import io.ksmt.utils.getValue
import io.ksmt.utils.uncheckedCast
import java.io.*
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicLong
import kotlin.io.path.isRegularFile
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

fun serialize(ctx: KContext, expressions: List<KExpr<KBoolSort>>, outputStream: OutputStream) {
    val serializationCtx = AstSerializationCtx().apply { initCtx(ctx) }
    val marshaller = AstSerializationCtx.marshaller(serializationCtx)
    val emptyRdSerializationCtx = SerializationCtx(Serializers())

    val buffer = UnsafeBuffer(ByteArray(100_000))

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

class LogSolver<C : KSolverConfiguration>(
    private val ctx: KContext, private val baseSolver: KSolver<C>
) : KSolver<C> by baseSolver {

    companion object {
        val counter = AtomicLong(0)
    }

    init {
        File("formulas").mkdirs()
    }

    private fun getNewFileCounter(): Long {
        return synchronized(counter) {
            counter.getAndIncrement()
        }
    }

    private fun getNewFileName(): String {
        return "formulas/f-${getNewFileCounter()}.bin"
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
        serialize(ctx, stack.flatten(), FileOutputStream(getNewFileName()))
        return baseSolver.check(timeout)
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        serialize(ctx, stack.flatten() + assumptions, FileOutputStream(getNewFileName()))
        return baseSolver.checkWithAssumptions(assumptions, timeout)
    }
}

fun main() {
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
    }

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