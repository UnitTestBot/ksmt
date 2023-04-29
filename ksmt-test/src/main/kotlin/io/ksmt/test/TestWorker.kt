package io.ksmt.test

import com.jetbrains.rd.framework.IProtocol
import io.ksmt.runner.core.KsmtWorkerBase
import io.ksmt.runner.core.RdServer
import io.ksmt.runner.generated.models.TestProtocolModel
import io.ksmt.runner.generated.models.testProtocolModel

class TestWorker(id: Int, process: RdServer) : KsmtWorkerBase<TestProtocolModel>(id, process) {
    override fun initProtocolModel(protocol: IProtocol): TestProtocolModel =
        protocol.testProtocolModel
}
