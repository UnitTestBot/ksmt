import io.ksmt.KContext
import io.ksmt.expr.*
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.neurosmt.KNeuroSMTSolver
import io.ksmt.solver.neurosmt.smt2converter.FormulaGraphExtractor
import io.ksmt.solver.neurosmt.smt2converter.getAnswerForTest
import io.ksmt.solver.z3.*
import io.ksmt.sort.*
import io.ksmt.utils.getValue
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isRegularFile
import kotlin.io.path.name
import kotlin.time.Duration.Companion.seconds

fun lol(a: Any) {
    if (a is KConst<*>) {
        println("const: ${a.decl.name}")
    }
    if (a is KInterpretedValue<*>) {
        println("val: ${a.decl.name}")
    }
}

class Kek(override val ctx: KContext) : KNonRecursiveTransformer(ctx) {
    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> {
        println("===")
        //println(expr::class)
        //lol(expr)
        //println(expr)
        /*
        println(expr.args)
        println("${expr.decl} ${expr.decl.name} ${expr.decl.argSorts} ${expr.sort}")
        for (child in expr.args) {
            println((child as KApp<*, *>).decl.name)
        }
        */
        return expr
    }
}

fun main() {
    val ctx = KContext()

    with(ctx) {
        val files = Files.walk(Path.of("../../neurosmt-benchmark/non-incremental/QF_BV")).filter { it.isRegularFile() }
        var ok = 0; var fail = 0
        var sat = 0; var unsat = 0; var unk = 0

        files.forEach {
            println(it)
            if (it.name.endsWith(".txt")) {
                return@forEach
            }

            when (getAnswerForTest(it)) {
                KSolverStatus.SAT -> sat++
                KSolverStatus.UNSAT -> unsat++
                KSolverStatus.UNKNOWN -> unk++
            }

            return@forEach

            val formula = try {
                val assertList = KZ3SMTLibParser(this).parse(it)
                ok++
                mkAnd(assertList)
            } catch (e: Exception) {
                fail++
                return@forEach
            }

            val extractor = FormulaGraphExtractor(this, formula, FileOutputStream("kek2.txt"))
            extractor.extractGraph()

            println("$ok : $fail")
        }

        println("$sat/$unsat/$unk")

        return@with

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
                (mkBvMulExpr(e, e) eq mkBv(9.toByte())) // and (mkExistentialQuantifier(x eq 2.expr, listOf()))
                //(mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardZero), h, h) eq 2.0.expr)

        //val constraint = mkBvXorExpr(mkBvShiftLeftExpr(e, mkBv(1.toByte())), mkBvNotExpr(e)) eq
        //        mkBvLogicalShiftRightExpr(e, mkBv(1.toByte()))

        // (constraint as KAndExpr)

        //constraint.accept(Kek(this))
        //Kek(this).apply(constraint)
        val extractor = FormulaGraphExtractor(this, constraint, FileOutputStream("kek.txt"))
        //val extractor = io.ksmt.solver.neurosmt.preprocessing.FormulaGraphExtractor(this, constraint, System.err)
        extractor.extractGraph()

        /*println("========")
        constraint.apply {
            println(this.args)
        }*/

        return@with

        KNeuroSMTSolver(this).use { solver -> // create a Stub SMT solver instance
            // assert expression
            solver.assert(constraint)

            // check assertions satisfiability with timeout
            val satisfiability = solver.check(timeout = 1.seconds)
            // println(satisfiability) // SAT
        }

        KZ3Solver(this).use { solver ->
            solver.assert(constraint)

            val satisfiability = solver.check(timeout = 180.seconds)
            println(satisfiability)

            if (satisfiability == KSolverStatus.SAT) {
                val model = solver.model()
                println(model)
            }
        }

        val formula = """
            (declare-fun x () Real)
            (declare-fun y () Real)
            (declare-fun z () Real)
            (assert (or (and (= y (+ x z)) (= x (+ y z))) (= 2.71 x)))
            (check-sat)
        """
        val assertions = KZ3SMTLibParser(this).parse(formula)

        println(assertions)
        println(assertions[0].stringRepr)
    }
}