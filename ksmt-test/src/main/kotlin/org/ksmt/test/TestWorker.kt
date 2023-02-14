package org.ksmt.test

import com.jetbrains.rd.framework.IProtocol
import org.ksmt.runner.core.KsmtWorkerBase
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.generated.models.TestProtocolModel
import org.ksmt.runner.generated.models.testProtocolModel

class TestWorker(id: Int, process: RdServer) : KsmtWorkerBase<TestProtocolModel>(id, process) {
    override fun initProtocolModel(protocol: IProtocol): TestProtocolModel =
        protocol.testProtocolModel
}
