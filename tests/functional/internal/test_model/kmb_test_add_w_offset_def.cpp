//
// Copyright 2021 Intel Corporation.
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

#include "kmb_test_add_w_offset_def.hpp"

#include <blob_transform.hpp>

namespace {

static void refAddWOffsetFromVPU(const Blob::Ptr src1, const Blob::Ptr src2, Blob::Ptr dst, float offset) {
    IE_ASSERT(src1 != nullptr);
    IE_ASSERT(src2 != nullptr);
    IE_ASSERT(dst != nullptr);

    const auto src1Data = src1->buffer().as<const float*>();
    const auto src2Data = src2->buffer().as<const float*>();
    const auto dstData = dst->buffer().as<float*>();
    IE_ASSERT(src1Data != nullptr);
    IE_ASSERT(src1Data != nullptr);
    IE_ASSERT(dstData != nullptr);

    const auto& dims = src1->getTensorDesc().getDims();
    IE_ASSERT(dims[0] == 1);
    const int IC = dims[1];
    const int IH = dims[2];
    const int IW = dims[3];

    for (int i = 0; i < IH*IW*IC; i++) {
        dstData[i] = src1Data[i] + src2Data[i] + offset;
    }
}

BlobVector refAddWOffset(const TestNetwork::NodePtr& layer, const BlobVector& inputs, const TestNetwork&) {
    IE_ASSERT(layer != nullptr);
    IE_ASSERT(inputs.size() == 2);

    const auto addWOffsetLayer = std::dynamic_pointer_cast<SampleExtension::AddWOffsetOp>(layer);

    IE_ASSERT(addWOffsetLayer != nullptr);

    const auto offset = addWOffsetLayer->getOffsetAttr();

    auto input1 = inputs.at(0);
    auto input2 = inputs.at(1);
    auto output = vpux::makeSplatBlob(input1->getTensorDesc(), 0.0f);

    refAddWOffsetFromVPU(input1, input2, output, offset);

    return {output};
}

}  // namespace

TestNetwork& AddWOffsetLayerDef::build() {
    std::shared_ptr<SampleExtension::AddWOffsetOp> addWOffsetNode =
        std::make_shared<SampleExtension::AddWOffsetOp>(
            testNet.getPort(input1Port), testNet.getPort(input2Port), params.offset);

    return testNet.addLayer(name, addWOffsetNode, refAddWOffset);
}
