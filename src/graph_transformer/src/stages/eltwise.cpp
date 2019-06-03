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
#include <string>
#include <unordered_set>
#include <memory>
#include <set>
#include <map>
#include <limits>
#include <algorithm>

#include <vpu/utils/numeric.hpp>

#define MAP_ELEMENTS(op, f) {InferenceEngine::EltwiseLayer::eOperation::op, &f<StageType::op>}

namespace vpu {

namespace {

template<StageType T>
StageType onlyOneInput(ie::EltwiseLayer::eOperation op, size_t input_size) {
    if (input_size != 1) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports only one input";
    }
    return T;
}

template<StageType T>
StageType onlyTwoInputs(ie::EltwiseLayer::eOperation op, size_t input_size) {
    if (input_size != 2) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports only two inputs";
    }
    return T;
}

template<StageType T>
StageType moreThanOneInput(ie::EltwiseLayer::eOperation op, size_t input_size) {
    if (input_size < 2) {
        VPU_THROW_EXCEPTION << "Eltwise operation: " << T << " supports two inputs and more";
    }
    return T;
}

const std::map<ie::EltwiseLayer::eOperation, std::function<StageType(ie::EltwiseLayer::eOperation, size_t)>> eltwise_map = {
        MAP_ELEMENTS(Sum,           moreThanOneInput),
        MAP_ELEMENTS(Prod,          moreThanOneInput),
        MAP_ELEMENTS(Max,           moreThanOneInput),
        MAP_ELEMENTS(Div,           onlyTwoInputs),
        MAP_ELEMENTS(Min,           moreThanOneInput),
        MAP_ELEMENTS(Squared_diff,  onlyTwoInputs),
        MAP_ELEMENTS(Equal,         onlyTwoInputs),
        MAP_ELEMENTS(Not_equal,     onlyTwoInputs),
        MAP_ELEMENTS(Greater,       onlyTwoInputs),
        MAP_ELEMENTS(Greater_equal, onlyTwoInputs),
        MAP_ELEMENTS(Less,          onlyTwoInputs),
        MAP_ELEMENTS(Less_equal,    onlyTwoInputs),
        MAP_ELEMENTS(Logical_NOT,   onlyOneInput),
        MAP_ELEMENTS(Logical_AND,   moreThanOneInput),
        MAP_ELEMENTS(Logical_OR,    moreThanOneInput),
        MAP_ELEMENTS(Logical_XOR,   moreThanOneInput),
        MAP_ELEMENTS(Pow,           onlyTwoInputs),
        MAP_ELEMENTS(Floor_mod,     onlyTwoInputs),
};

class EltwiseStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<EltwiseStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto output = _outputEdges[0]->output();

        if (_type != StageType::Prod &&
            step == ScalePropagationStep::Propagate) {
            // Keep the largest input scale factor.
            auto maxScale = std::numeric_limits<float>::lowest();
            for (const auto& inEdge : _inputEdges) {
                maxScale = std::max(maxScale, inputScales[inEdge->portInd()]);
            }

            for (const auto& inEdge : _inputEdges) {
                auto curScale = inputScales[inEdge->portInd()];

                if (!isFloatEqual(curScale, maxScale)) {
                    _scaleInfo.setInput(inEdge, maxScale / curScale);
                }
            }

            _scaleInfo.setOutput(_outputEdges[0], maxScale);
        } else {
            // Eltwise can only propagate scaling for Sum and Max cases.
            for (const auto& inEdge : _inputEdges) {
                _scaleInfo.setInput(inEdge, 1.0f);
            }

            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        }
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        auto in0Desc = input0->desc();
        auto in1Desc = input1->desc();
        auto outDesc = output->desc();

        auto finalOrder  = in0Desc.numDims() >= in1Desc.numDims() ? in0Desc.dimsOrder() : in1Desc.dimsOrder();
        auto secondOrder = in0Desc.numDims() >= in1Desc.numDims() ? in1Desc.dimsOrder() : in0Desc.dimsOrder();
        if (secondOrder.numDims() >= 3) {
            if (secondOrder.dimInd(Dim::C) == 1 /*HCW*/) {
                finalOrder = secondOrder;
            } else if (secondOrder.dimInd(Dim::C) == 2 /*CHW*/ && finalOrder.dimInd(Dim::C) != 1 /*HCW*/) {
                finalOrder = secondOrder;
            }
        }
        if (outDesc.numDims() > finalOrder.numDims()) {
            finalOrder = outDesc.dimsOrder();
        }

        _orderInfo.setInput(_inputEdges[0], finalOrder.numDims() == in0Desc.numDims() ? finalOrder : in0Desc.dimsOrder());
        _orderInfo.setInput(_inputEdges[1], finalOrder.numDims() == in1Desc.numDims() ? finalOrder : in1Desc.dimsOrder());
        _orderInfo.setOutput(_outputEdges[0], finalOrder);
    }

    void getDataStridesRequirementsImpl() const override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto coeff1 = attrs().getOrDefault<float>("coeff1", 1.0f);
        auto coeff2 = attrs().getOrDefault<float>("coeff2", 1.0f);

        serializer.append(static_cast<float>(coeff1));
        serializer.append(static_cast<float>(coeff2));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 2);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        input0->serializeNewBuffer(serializer, output->desc().dimsOrder());
        output->serializeNewBuffer(serializer);
        input1->serializeNewBuffer(serializer, output->desc().dimsOrder());
    }
};

}  // namespace

void FrontEnd::parseEltwise(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    auto layer = std::dynamic_pointer_cast<ie::EltwiseLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    IE_ASSERT(outputs.size() == 1);

    auto stageType = StageType::None;
    auto subCoefficient = 1.0f;

    if (layer->_operation == ie::EltwiseLayer::eOperation::Sub) {
        if (inputs.size() != 2) {
            VPU_THROW_EXCEPTION << "Eltwise operation: " << layer->_operation << " with multiple inputs is not supported";
        }
        stageType = StageType::Sum;
        subCoefficient = -1.f;
    } else if (layer->_operation == ie::EltwiseLayer::eOperation::Mean) {
        if (inputs.size() != 2) {
            VPU_THROW_EXCEPTION << "Eltwise operation: " << layer->_operation << " with multiple inputs is not supported";
        }
        stageType = StageType::Sum;
    } else {
        if (eltwise_map.find(layer->_operation) != eltwise_map.end()) {
            stageType = eltwise_map.at(layer->_operation)(layer->_operation, inputs.size());
        } else {
            VPU_THROW_EXCEPTION << "Eltwise operation: " << layer->_operation << " is not supported";
        }
    }

    if (stageType != StageType::Sum && !layer->coeff.empty()) {
        VPU_THROW_EXCEPTION << layer->name << " coefficients only supported for Sum/Sub operations.";
    }

    auto output = outputs[0];

    auto tempOutput = output;
    if (inputs.size() > 2) {
        tempOutput = model->duplicateData(
            output,
            formatString("@temp@1/%d", inputs.size() - 2));
    }

    DataVector tempInputs(2);
    tempInputs[0] = inputs[0];

    if (stageType == StageType::Logical_NOT)
        tempInputs[1] = model->addFakeData();
    else
        tempInputs[1] = inputs[1];

    auto stage = model->addNewStage<EltwiseStage>(
        layer->name,
        stageType,
        layer,
        tempInputs,
        {tempOutput});

    if (layer->_operation == ie::EltwiseLayer::eOperation::Mean) {
        stage->attrs().set<float>("coeff1",  0.5);
        stage->attrs().set<float>("coeff2",  0.5);
    } else {
        if (layer->coeff.size() > 0) {
            stage->attrs().set<float>("coeff1", layer->coeff[0]);
        }
        if (layer->coeff.size() > 1 || subCoefficient != 1.0f) {
            stage->attrs().set<float>("coeff2", subCoefficient * (layer->coeff.size() > 1 ? layer->coeff[1] : 1.0f));
        }
    }

    tempInputs[0] = tempOutput;
    for (int ind = 2; ind < inputs.size(); ++ind) {
        tempInputs[1] = inputs[ind];

        if (ind + 1 == inputs.size()) {
            tempOutput = output;
        } else {
            tempOutput = model->duplicateData(
                output,
                formatString("@temp@%d/%d", ind, inputs.size() - 2));
        }

        stage = model->addNewStage<EltwiseStage>(
            layer->name + "@" + std::to_string(ind - 1),
            stageType,
            layer,
            tempInputs,
            {tempOutput});

        if (layer->coeff.size() > ind) {
            stage->attrs().set<float>("coeff2", layer->coeff[ind]);
        }

        tempInputs[0] = tempOutput;
    }
}

Stage StageBuilder::addSumStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input0,
        const Data& input1,
        const Data& output) {
    return model->addNewStage<EltwiseStage>(
        name,
        StageType::Sum,
        layer,
        {input0, input1},
        {output});
}

}  // namespace vpu
