//
// Copyright 2020 Intel Corporation.
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

#include "test_model/kmb_test_base.hpp"

#include <vpux/utils/IE/format.hpp>

struct ConvertTestParams final {
    ConvertParams params;

    LAYER_PARAMETER(ngraph::element::Type, destination_type);
    PARAMETER(ngraph::element::Type, source_type);
    PARAMETER(SizeVector, dims);
};

std::ostream& operator<<(std::ostream& os, const ConvertTestParams &p) {
    vpux::printTo(
        os, "dims: {0}, source_type: {1}, destination_type: {2}", p.dims(), p.source_type(), p.destination_type());
    return os;
}

class KmbConvertLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<ConvertTestParams> {};

// After migrating to ngraph parser, need to add a way of disabling other covert stages/passes
// [Track number: S#41542]
// Disabling reason: with the new input precision support, the first tensor has fp16 data type and is not
// converted properly by this kernel; ConfigureIOPrecision would also add an extra Conversion op from u8->fp16
TEST_P(KmbConvertLayerTests, DISABLED_accuracy) {
    const auto &p = GetParam();

    const auto netPrecision = Precision::FP32;

    const auto dims = p.dims();
    const auto userInDesc = TensorDesc(typeToPrecision(p.source_type()), dims, Layout::NCHW);
    const auto userOutDesc = TensorDesc(typeToPrecision(p.destination_type()), dims, Layout::NCHW);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
    });

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPrecision)
            .addLayer<ConvertLayerDef>("convert", p.params)
                .input("input")
                .build()
            .addNetOutput(PortInfo("convert"))
            .setUserOutput(PortInfo("convert"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .useCustomLayers(KernelType::Ocl)
            .disableMcmPasses({{"kmb_adapt", "KMBQuantizeConversion"}})
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<ConvertTestParams> convertParams = {
        ConvertTestParams()
            .dims({1, 3, 360, 480})
            .source_type(ngraph::element::Type_t::u8)
            .destination_type(ngraph::element::Type_t::f16)
};

#ifdef KMB_HAS_CUSTOM_OCL_KERNELS
INSTANTIATE_TEST_SUITE_P(precommit, KmbConvertLayerTests, testing::ValuesIn(convertParams));
#endif
