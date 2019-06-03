//
// Copyright 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>
#include <string>

#include <vpu/sw/post_op_stage.hpp>

namespace vpu {

namespace {

class ScaleStage final : public PostOpStage {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ScaleStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 2 || _inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto inputScale = inputScales[0];

        _scaleInfo.setInput(_inputEdges[1], step == ScalePropagationStep::Propagate ? 1.0f : inputScale);
        if (_inputEdges.size() == 3) {
            _scaleInfo.setInput(_inputEdges[2], inputScale);
        }
        _scaleInfo.setOutput(_outputEdges[0], inputScale);
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }
};

}  // namespace

Stage StageBuilder::addScaleStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& scales,
        const Data& output) {
    return model->addNewStage<ScaleStage>(
        name,
        StageType::Scale,
        layer,
        {input, scales},
        {output});
}

void FrontEnd::parseScale(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::ScaleShiftLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    if (layer->_broadcast != 0) {
        VPU_THROW_EXCEPTION <<
            "Layer " << layer->name << " doesn't support broadcast param";
    }

    auto input = inputs[0];
    auto output = outputs[0];

    Data scales, biases;
    std::tie(scales, biases) = getWeightsAndBiases(model, layer);

    if (biases->usage() == DataUsage::Fake) {
        model->addNewStage<ScaleStage>(
            layer->name,
            StageType::Scale,
            layer,
            {input, scales},
            {output});
    } else {
        model->addNewStage<ScaleStage>(
            layer->name,
            StageType::ScaleShift,
            layer,
            {input, scales, biases},
            {output});
    }
}

}  // namespace vpu
