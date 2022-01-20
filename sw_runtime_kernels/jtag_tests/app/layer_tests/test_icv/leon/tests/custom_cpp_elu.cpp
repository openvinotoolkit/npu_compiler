//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <custom_cpp_tests.h>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.elu_fp16.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_elu.h"

// To workaround integer array 'layerParams'.
union Hex {
    float f;
    int32_t i;
};

#define F_0_5 0x3f000000           // Hex representation of 0.5f
#define USE_SEED_VALUE 0xbdd1cb13  // defined to use this value as random seed

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Elu)) {
    static constexpr std::initializer_list<SingleTest> elu_test_list{
            {{4, 4, 4},
             {4, 4, 4},
             orderZYX,
             FPE("elu.elf"),
             {{
                     F_0_5 /*alpha*/,
                     sw_params::Location::NN_CMX /*mem type*/,
             }}},
            {{2, 3, 5},
             {2, 3, 5},
             orderZYX,
             FPE("elu.elf"),
             {{
                     F_0_5 /*alpha*/,
                     sw_params::Location::NN_CMX /*mem type*/,
             }}},
    };

    class CustomCppEluTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppEluTest(): m_testsLoop(elu_test_list, "test") {
        }
        virtual ~CustomCppEluTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppEluTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};

            paramContainer.resize(((int)sizeof(sw_params::EluParams) + 7) / 8);
            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            const Hex alpha_hex = {.i = test->customLayerParams.layerParams[0]};
            m_alpha = alpha_hex.f;
            m_eluParams = reinterpret_cast<sw_params::EluParams*>(paramContainer.data());
            *m_eluParams = sw_params::EluParams();
            m_eluParams->alpha = m_alpha;
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer.data());
            m_params.paramDataLen = paramContainer.size() * sizeof(uint64_t);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[1]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_eluParams);

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(sk_elu_fp16_3010xx_text);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(elu_fp16));
#endif
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.008f;
        }

        void generateInputData() override {
#if defined(USE_SEED_VALUE)
            auto seedValue = USE_SEED_VALUE;
#else
            u64 systemTicks;
            DrvTimerGetSystemTicks64(&systemTicks);
            auto seedValue = static_cast<unsigned int>(systemTicks);
#endif
            std::mt19937 generator(seedValue);
            m_inputTensor.forEach(false, [this, &generator](const MemoryDims& indices) {
                // We generate the random value between -8.f and 8f and the kernel do x * relu6(x+3) / 6
                // So the minimum resolution is 2^(-7) = 0.00781f and the kernel may calculate 0 output value
                // if input value is less than this resolution. In such cases, relative difference would be 1.
                const float precisionLimitations = 0.00781f;
                float fp32Value = 0.f;
                do {
                    fp32Value = float(generator()) / generator.max() * 16.f - 8.f;
                } while (fabs(fp32Value) < precisionLimitations && fp32Value != 0.f);

                m_inputTensor.at(indices) = f32Tof16(fp32Value);
            });
            // reference output
            generateReferenceData();
        }
        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float min = std::min(val, 0.0f);
                float max = std::max(val, 0.0f);
                float ref = max + m_alpha * (exp((double)min) - 1.0f);
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }
        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");
            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= m_test_threshold);

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value,
                           abs_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;

        // Additional buffer to avoid convertion back and forth
        float m_alpha;
        std::vector<uint64_t> paramContainer;
        sw_params::EluParams* m_eluParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppEluTest)
}  // namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME,Elu))
