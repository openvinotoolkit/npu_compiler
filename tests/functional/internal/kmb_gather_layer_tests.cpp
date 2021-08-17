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
#include <blob_factory.hpp>

struct GatherTestParams final {
    SizeVector _dataDims;
    Precision _dataPrecision;
    Layout _dataLayout;

    SizeVector _indicesDims;
    Layout _indicesLayout;

    SizeVector _axisDims;

    size_t _axis;
    std::vector<size_t> _indices;

    GatherTestParams& dataDims(const SizeVector& dataDims) {
        this->_dataDims = dataDims;
        return *this;
    }
    GatherTestParams& dataPrecision(const Precision& dataPrecision) {
        this->_dataPrecision = dataPrecision;
        return *this;
    }
    GatherTestParams& dataLayout(const Layout& dataLayout) {
        this->_dataLayout = dataLayout;
        return *this;
    }

    GatherTestParams& indicesDims(const SizeVector& indicesDims) {
        this->_indicesDims = indicesDims;
        return *this;
    }
    GatherTestParams& indicesLayout(const Layout& indicesLayout) {
        this->_indicesLayout = indicesLayout;
        return *this;
    }

    GatherTestParams& axisDims(const SizeVector& axisDims) {
        this->_axisDims = axisDims;
        return *this;
    }

    GatherTestParams& axis(size_t axis) {
        this->_axis = axis;
        return *this;
    }
    GatherTestParams& indices(const std::vector<size_t>& indices) {
        this->_indices = indices;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const GatherTestParams& p) {
    vpu::formatPrint(os, "[axis:%v, indices size:%v]", p._axis, p._indices.size());
    return os;
}

Blob::Ptr makeBlobFromData(const TensorDesc& desc, std::vector<size_t> data) {
    size_t blobElements = 1;
    std::for_each(desc.getDims().begin(), desc.getDims().end(), [&blobElements](size_t element) {
        blobElements *= element;
    });
    assert(blobElements == data.size());
    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    const auto outPtr = blob->buffer().as<int*>();
    std::copy(data.data(), data.data() + data.size(), outPtr);
    return blob;
}

class internal_KmbGatherLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<GatherTestParams> {};

TEST_P(internal_KmbGatherLayerTests, EqualWithCPU) {
    // TODO: Need to fix bad check in gather layer parser in runtime
    SKIP_INFER("Hangs on runtime");
    const auto &p = GetParam();

    const auto userDataDesc = TensorDesc(p._dataPrecision, p._dataDims, p._dataLayout);
    const auto userIndicesDesc = TensorDesc(Precision::I32, p._indicesDims, p._indicesLayout);
    const auto userAxisDesc = TensorDesc(Precision::I64, {1}, Layout::C);
    const auto userOutDesc = TensorDesc(p._dataPrecision, p._dataLayout);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-3f;

    registerBlobGenerator(
        "input", userDataDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );

    std::vector<float> indicesArray{};

    registerBlobGenerator(
        "indices", userIndicesDesc,
        [&](const TensorDesc& desc) {
            return makeBlobFromData(desc, p._indices);
        }
    );
    registerBlobGenerator(
        "axis", userAxisDesc,
        [&](const TensorDesc& desc) {
            return vpux::makeSplatBlob(desc, float(p._axis));
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userDataDesc.getPrecision(), userDataDesc.getLayout())
            .addNetInput("input", userDataDesc.getDims(), userDataDesc.getPrecision())
            .addConst("indices", getBlobByName("indices"))
            .addConst("axis", getBlobByName("axis"))
            .addLayer<GatherLayerDef>("gather")
                .input("input")
                .indices("indices")
                .axis("axis")
                .build()
            .addNetOutput(PortInfo("gather"))
            .setUserOutput(PortInfo("gather"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

// Params from ICNet network
const std::vector<GatherTestParams> gatherParams {
        GatherTestParams()
            .dataDims({1, 14})
                .dataLayout(Layout::NC)
                .dataPrecision(Precision::FP16)
            .indicesDims({7})
                .indicesLayout(Layout::C)
                .indices({1, 3, 5, 7, 9, 11, 13})
            .axisDims({1})
                .axis(1),
        GatherTestParams()
            .dataDims({1,2})
                .dataLayout(Layout::NC)
                .dataPrecision(Precision::FP32)
            .indicesDims({1})
                .indicesLayout(Layout::C)
                .indices({0})
            .axisDims({1})
                .axis(1),
        GatherTestParams()
            .dataDims({2})
                .dataLayout(Layout::C)
                .dataPrecision(Precision::FP32)
            .indicesDims({1})
                .indicesLayout(Layout::C)
                .indices({1})
            .axisDims({1})
                .axis(0)
};

INSTANTIATE_TEST_SUITE_P(Gather, internal_KmbGatherLayerTests, testing::ValuesIn(gatherParams));
