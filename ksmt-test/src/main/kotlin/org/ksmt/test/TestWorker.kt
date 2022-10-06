package org.ksmt.test

import org.ksmt.runner.core.KsmtWorkerBase
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.generated.TestProtocolModel
import org.ksmt.runner.generated.testProtocolModel

class TestWorker(id: Int, process: RdServer) : KsmtWorkerBase<TestProtocolModel>(id, process) {
    override val protocolModel: TestProtocolModel
        get() = protocol.testProtocolModel
}
