package io.ksmt.solver.runner

import com.jetbrains.rd.framework.IProtocol
import io.ksmt.runner.core.KsmtWorkerBase
import io.ksmt.runner.core.RdServer
import io.ksmt.runner.generated.models.SolverProtocolModel
import io.ksmt.runner.generated.models.solverProtocolModel

class KSolverWorker(id: Int, process: RdServer) : KsmtWorkerBase<SolverProtocolModel>(id, process) {
    override fun initProtocolModel(protocol: IProtocol): SolverProtocolModel =
        protocol.solverProtocolModel
}
