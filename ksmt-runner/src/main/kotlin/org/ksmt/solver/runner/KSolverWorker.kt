package org.ksmt.solver.runner

import com.jetbrains.rd.framework.IProtocol
import org.ksmt.runner.core.KsmtWorkerBase
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.generated.models.SolverProtocolModel
import org.ksmt.runner.generated.models.solverProtocolModel

class KSolverWorker(id: Int, process: RdServer) : KsmtWorkerBase<SolverProtocolModel>(id, process) {
    override fun initProtocolModel(protocol: IProtocol): SolverProtocolModel =
        protocol.solverProtocolModel
}
