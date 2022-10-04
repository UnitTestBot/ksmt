package org.ksmt.solver.runner

import org.ksmt.runner.core.KsmtWorkerBase
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.generated.SolverProtocolModel
import org.ksmt.runner.generated.solverProtocolModel

class KSolverWorker(id: Int, process: RdServer) : KsmtWorkerBase<SolverProtocolModel>(id, process) {
    override val protocolModel: SolverProtocolModel
        get() = protocol.solverProtocolModel
}
