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

void FrontEnd::parsePower(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];

    auto layer = std::dynamic_pointer_cast<ie::PowerLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    _stageBuilder->addPowerStage(
        model,
        layer->name,
        layer,
        layer->scale,
        layer->power,
        layer->offset,
        inputs[0],
        outputs[0]);
}

namespace {

class PowerStage final : public PostOpStage {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PowerStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto power = attrs().get<float>("power");
        auto& scale = attrs().get<float>("scale");
        auto& bias = attrs().get<float>("bias");

        if (power != 1.0f) {
            _scaleInfo.setInput(_inputEdges[0], 1.0f);
            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        } else {
            auto inputScale = inputScales[0];

            _scaleInfo.setOutput(_outputEdges[0], inputScale);

            if (step == ScalePropagationStep::ScaleInput) {
                scale *= inputScale;
            }
            if (step != ScalePropagationStep::Check) {
                bias *= inputScale;
            }
        }
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto scale = attrs().get<float>("scale");
        auto power = attrs().get<float>("power");
        auto bias = attrs().get<float>("bias");

        serializer.append(static_cast<float>(bias));
        serializer.append(static_cast<float>(scale));
        serializer.append(static_cast<float>(power));
    }
};

}  // namespace

Stage StageBuilder::addPowerStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        float scale,
        float power,
        float bias,
        const Data& input,
        const Data& output) {
    auto stage = model->addNewStage<PowerStage>(
        name,
        StageType::Power,
        layer,
        {input},
        {output});

    stage->attrs().set<float>("scale", scale);
    stage->attrs().set<float>("power", power);
    stage->attrs().set<float>("bias", bias);

    return stage;
}

}  // namespace vpu
