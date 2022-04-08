//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <custom_cpp_tests.h>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.swish_fp16.3720xx.text.xdat"

#include "param_swish.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Swish)) {
    struct SwishCustomParams {
        float beta;
        int32_t layerParams[MAX_LOCAL_PARAMS];
    };

    struct SwishTest {
        Dims inputDims;
        Dims outputDims;
        StorageOrder storageOrder;
        SwishCustomParams customLayerParams;
    };

    static constexpr std::initializer_list<SwishTest> swish_test_list{
            {{1, 1, 10}, {1, 1, 10}, orderZYX, {1.3f /*beta*/, sw_params::Location::NN_CMX}},
            {{2, 20, 2}, {2, 20, 2}, orderZYX, {0.5f /*beta*/, sw_params::Location::NN_CMX}},
#ifdef CONFIG_RUN_LARGE_TESTS
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, {1.0f /*beta*/, sw_params::Location::NN_CMX}}
#endif
    };

    class CustomCppSwishTest : public CustomCppTests<fp16, SwishTest> {
    public:
        explicit CustomCppSwishTest(): m_testsLoop(swish_test_list, "test") {
        }
        virtual ~CustomCppSwishTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppSwishTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            CustomCppTests<fp16, SwishTest>::initData();
            const SwishTest* test = m_currentTest;

            m_swishParams = reinterpret_cast<sw_params::SwishParams*>(paramContainer);
            *m_swishParams = sw_params::SwishParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::SwishParams);
            m_beta = test->customLayerParams.beta;
            m_swishParams->beta = m_beta;
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_swishParams);

            m_params.kernel = reinterpret_cast<uint32_t>(sk_swish_fp16_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.015f;
        }

        void generateInputData() override {
            rand_seed();

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }

        void generateReferenceData() override {
            // no need to remap memory indices between tensors
            mvTensorAssert(m_inputTensor.storageOrder() == m_referenceOutputTensor.storageOrder());
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = val / (1.0f + exp(-m_swishParams->beta * val));
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();
            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float input = f16Tof32(m_inputTensor.at(indices));

                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= m_test_threshold);

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] input %f output %f ref %f diff %f\n", ti.height, ti.width, ti.channels,
                           input, value, gt_value, abs_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<SwishTest> m_testsLoop;
        sw_params::SwishParams* m_swishParams;
        float m_beta;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppSwishTest)
}  // namespace )
