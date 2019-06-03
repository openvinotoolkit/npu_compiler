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

#include <cmath>

#include <vector>
#include <limits>
#include <memory>
#include <set>
#include <string>

#include <vpu/sw/post_op_stage.hpp>

namespace vpu {

namespace {

class ReLUStage final : public PostOpStage {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ReLUStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        if (step == ScalePropagationStep::Propagate) {
            auto inputScale = inputScales[0];
            _scaleInfo.setOutput(_outputEdges[0], inputScale);
        } else {
            // ReLU can only propagate scaling, not generate.
            _scaleInfo.setInput(_inputEdges[0], 1.0f);
            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        }
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto negativeSlope = attrs().get<float>("negativeSlope");

        serializer.append(static_cast<uint32_t>(_inputEdges.size() == 2));
        serializer.append(negativeSlope);
    }
};

}  // namespace

void FrontEnd::parseReLU(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::ReLULayer>(_layer);
    IE_ASSERT(layer != nullptr);

    _stageBuilder->addReLUStage(model, layer->name, layer, layer->negative_slope, inputs[0], outputs[0]);
}

Stage StageBuilder::addReLUStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        float negativeSlope,
        const Data& input,
        const Data& output,
        const Data& biases) {
    auto stageType = StageType::__SPECIAL_START__;
    if (biases == nullptr) {
        stageType =
            std::fabs(negativeSlope) < std::numeric_limits<float>::epsilon() ?
                StageType::Relu :
                StageType::LeakyRelu;
    } else {
        stageType =
            std::fabs(negativeSlope) < std::numeric_limits<float>::epsilon() ?
                StageType::BiasRelu :
                StageType::BiasLeakyRelu;
    }

    auto stage = model->addNewStage<ReLUStage>(
        name,
        stageType,
        layer,
        {input},
        {output});

    if (biases != nullptr) {
        model->addStageInput(stage, biases);
    }

    stage->attrs().set<float>("negativeSlope", negativeSlope);

    return stage;
}

}  // namespace vpu
