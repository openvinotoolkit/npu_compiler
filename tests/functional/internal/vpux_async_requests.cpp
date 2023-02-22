//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "test_model/kmb_test_base.hpp"

#include <condition_variable>
#include <mutex>
#include <string>

struct AsyncTestParams final {
    PARAMETER(std::size_t, nireq);
    PARAMETER(std::size_t, niter);
};

std::ostream& operator<<(std::ostream& os, const AsyncTestParams& p) {
    vpux::printTo(os, "nireq: {0}, niter: {1}", p.nireq(), p.niter());
    return os;
}

class VpuxAsyncTests : public KmbLayerTestBase, public testing::WithParamInterface<AsyncTestParams> {};

// [Track number: E#14836]
TEST_P(VpuxAsyncTests, regression_ADK) {
    SKIP_ON("LEVEL0", "Bad results");
    SKIP_ON("EMULATOR", "Export network not enabled on emulator.");
    const auto& p = GetParam();
    const std::size_t nireq = p.nireq();
    const std::size_t niter = p.niter();

    // for VPUX30XX/VPU311X standalone (VPUAL backend): there is a data race, the test sporadicaly fails: S#49626
    // TODO: it makes sense to introduce a separate macro for such SKIP
    SKIP_INFER_ON("VPUAL", "data race");

    if (RUN_COMPILER) {
        const std::vector<size_t> dims = {1, 3, 32, 32};
        const auto layout = Layout::NHWC;
        const auto userInDesc = TensorDesc(Precision::U8, dims, layout);
        const auto userOutDesc = TensorDesc(Precision::FP16, layout);

        const auto scaleDesc = TensorDesc(Precision::FP32, dims, layout);
        registerBlobGenerator("scale", scaleDesc, [&](const TensorDesc& desc) {
            return vpux::makeSplatBlob(desc, 1.f);
        });

        for (std::size_t i = 0; i < nireq; i++) {
            registerBlobGenerator(std::string("input") + std::to_string(i), userInDesc, [&](const TensorDesc& desc) {
                return vpux::makeSplatBlob(desc, 1.f + static_cast<float>(i));
            });
        }

        const auto netPrecision = Precision::FP32;
        TestNetwork testNet;
        testNet.setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
                .addNetInput("input", userInDesc.getDims(), netPrecision)
                .addLayer<PowerLayerDef>("power")
                .input1("input")
                .input2(getBlobByName("scale"))
                .build()
                .setUserOutput(PortInfo("power"), userOutDesc.getPrecision(), userOutDesc.getLayout())
                .addNetOutput(PortInfo("power"))
                .finalize();

        auto exeNet = getExecNetwork(testNet);

        KmbTestBase::exportNetwork(exeNet);
    }

    if (RUN_INFER) {
        auto exeNet = KmbTestBase::importNetwork();
        std::atomic<std::size_t> iterationCount(0);
        std::condition_variable waitCompletion;
        std::vector<InferenceEngine::InferRequest> inferRequests(nireq);
        std::vector<Blob::Ptr> inputs(nireq);
        std::atomic<std::size_t> idleReqs(0);
        const auto& inputDesc = exeNet.GetInputsInfo().begin()->second->getTensorDesc();
        for (std::size_t i = 0; i < nireq; i++) {
            registerBlobGenerator(std::string("input") + std::to_string(i), inputDesc, [&](const TensorDesc& desc) {
                return vpux::makeSplatBlob(desc, 1.f + i);
            });
            inferRequests[i] = exeNet.CreateInferRequest();
            inputs[i] = KmbTestBase::getBlobByName(std::string("input") + std::to_string(i));
            inferRequests[i].SetBlob("input", inputs[i]);
            auto onComplete = [&waitCompletion, &iterationCount, &idleReqs, &inferRequests, i, &inputs, &niter,
                               this](void) -> void {
                iterationCount++;
                if (iterationCount < niter) {
                    Blob::Ptr output = inferRequests[i].GetBlob("power");
                    KmbTestBase::compareOutputs(inputs[i], output, 0, CompareMethod::Absolute);
                    inferRequests[i].StartAsync();
                } else {
                    idleReqs++;
                    waitCompletion.notify_one();
                }
            };

            inferRequests[i].SetCompletionCallback(onComplete);
        }

        for (std::size_t i = 0; i < nireq; i++) {
            inferRequests[i].StartAsync();
        }

        std::mutex execGuard;
        std::unique_lock<std::mutex> execLocker(execGuard);
        waitCompletion.wait(execLocker, [&] {
            return idleReqs == nireq;
        });
    }
}

const std::vector<AsyncTestParams> asyncParams = {
        AsyncTestParams().nireq(1).niter(100),
        AsyncTestParams().nireq(2).niter(100),
        AsyncTestParams().nireq(4).niter(100),
        AsyncTestParams().nireq(8).niter(100),
};

INSTANTIATE_TEST_SUITE_P(precommit, VpuxAsyncTests, testing::ValuesIn(asyncParams));
