package org.ksmt.solver.runner

import com.jetbrains.rd.framework.IProtocol
import org.ksmt.runner.core.KsmtWorkerBase
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.models.generated.SolverProtocolModel
import org.ksmt.runner.models.generated.solverProtocolModel

class KSolverWorker(id: Int, process: RdServer) : KsmtWorkerBase<SolverProtocolModel>(id, process) {
    override fun initProtocolModel(protocol: IProtocol): SolverProtocolModel =
        protocol.solverProtocolModel
}
