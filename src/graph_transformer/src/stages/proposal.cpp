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

#include <details/caseless.hpp>

namespace vpu {

namespace {

class ProposalStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ProposalStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();

        _orderInfo.setInput(_inputEdges[0], input0->desc().dimsOrder().createMovedDim(Dim::C, 2));
        _orderInfo.setInput(_inputEdges[1], input1->desc().dimsOrder().createMovedDim(Dim::C, 2));
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        _stridesInfo.setInput(_inputEdges[0], StridesRequirement::compact());
        _stridesInfo.setInput(_inputEdges[1], StridesRequirement::compact());
        _stridesInfo.setInput(_inputEdges[2], StridesRequirement::compact());
        _stridesInfo.setOutput(_outputEdges[0], StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto feat_stride = attrs().get<int>("feat_stride");
        auto base_size = attrs().get<int>("base_size");
        auto min_size = attrs().get<int>("min_size");
        auto pre_nms_topn = attrs().get<int>("pre_nms_topn");
        auto post_nms_topn = attrs().get<int>("post_nms_topn");
        auto nms_thresh = attrs().get<float>("nms_thresh");
        auto pre_nms_thresh = attrs().get<float>("pre_nms_thresh");
        auto box_size_scale = attrs().get<float>("box_size_scale");
        auto box_coordinate_scale = attrs().get<float>("box_coordinate_scale");
        auto coordinates_offset = attrs().get<float>("coordinates_offset");
        auto initial_clip = attrs().get<bool>("initial_clip");
        auto clip_before_nms = attrs().get<bool>("clip_before_nms");
        auto clip_after_nms = attrs().get<bool>("clip_after_nms");
        auto normalize = attrs().get<bool>("normalize");

        auto shift_anchors = attrs().get<bool>("shift_anchors");
        auto round_ratios = attrs().get<bool>("round_ratios");
        auto swap_xy = attrs().get<bool>("swap_xy");
        const auto& scales = attrs().get<std::vector<float>>("scales");
        const auto& ratios = attrs().get<std::vector<float>>("ratios");

        serializer.append(static_cast<uint32_t>(feat_stride));
        serializer.append(static_cast<uint32_t>(base_size));
        serializer.append(static_cast<uint32_t>(min_size));
        serializer.append(static_cast<int32_t>(pre_nms_topn));
        serializer.append(static_cast<int32_t>(post_nms_topn));
        serializer.append(static_cast<float>(nms_thresh));
        serializer.append(static_cast<float>(pre_nms_thresh));
        serializer.append(static_cast<float>(box_size_scale));
        serializer.append(static_cast<float>(box_coordinate_scale));
        serializer.append(static_cast<float>(coordinates_offset));
        serializer.append(static_cast<uint32_t>(initial_clip));
        serializer.append(static_cast<uint32_t>(clip_before_nms));
        serializer.append(static_cast<uint32_t>(clip_after_nms));
        serializer.append(static_cast<uint32_t>(normalize));
        serializer.append(static_cast<uint32_t>(shift_anchors));
        serializer.append(static_cast<uint32_t>(round_ratios));
        serializer.append(static_cast<uint32_t>(swap_xy));

        auto serializeVector = [&serializer](const std::vector<float>& array) {
            serializer.append(static_cast<uint32_t>(array.size()));
            for (auto elem : array) {
                serializer.append(static_cast<float>(elem));
            }
        };

        serializeVector(scales);
        serializeVector(ratios);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.size() == 1);

        auto input0 = _inputEdges[0]->input();
        auto input1 = _inputEdges[1]->input();
        auto input2 = _inputEdges[2]->input();
        auto output = _outputEdges[0]->output();

        input0->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
        input1->serializeNewBuffer(serializer);
        input2->serializeNewBuffer(serializer);
        _tempBufferEdges[0]->tempBuffer()->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseProposal(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    ie::details::CaselessEq<std::string> cmp;

    IE_ASSERT(inputs.size() == 3);
    IE_ASSERT(outputs.size() == 1);

    auto stage = model->addNewStage<ProposalStage>(
        layer->name,
        StageType::Proposal,
        layer,
        inputs,
        outputs);

    stage->attrs().set<int>("feat_stride", layer->GetParamAsInt("feat_stride", 16));
    stage->attrs().set<int>("base_size", layer->GetParamAsInt("base_size", 16));
    stage->attrs().set<int>("min_size", layer->GetParamAsInt("min_size", 16));
    stage->attrs().set<int>("pre_nms_topn", layer->GetParamAsInt("pre_nms_topn", 6000));
    stage->attrs().set<int>("post_nms_topn", layer->GetParamAsInt("post_nms_topn", 300));
    stage->attrs().set<float>("nms_thresh", layer->GetParamAsFloat("nms_thresh", 0.7f));
    stage->attrs().set<float>("pre_nms_thresh", layer->GetParamAsFloat("pre_nms_thresh", 0.1f));
    stage->attrs().set<float>("box_size_scale", layer->GetParamAsFloat("box_size_scale", 1.0f));
    stage->attrs().set<float>("box_coordinate_scale", layer->GetParamAsFloat("box_coordinate_scale", 1.0f));
    stage->attrs().set<bool>("clip_before_nms", layer->GetParamAsBool("clip_before_nms", true));
    stage->attrs().set<bool>("clip_after_nms", layer->GetParamAsBool("clip_after_nms", false));
    stage->attrs().set<bool>("normalize", layer->GetParamAsBool("normalize", false));

    if (cmp(layer->GetParamAsString("framework", ""), "TensorFlow")) {
        // Settings for TensorFlow
        stage->attrs().set<float>("coordinates_offset", 0.0f);
        stage->attrs().set<bool>("initial_clip", true);
        stage->attrs().set<bool>("shift_anchors", true);
        stage->attrs().set<bool>("round_ratios", false);
        stage->attrs().set<bool>("swap_xy", true);
    } else {
        // Settings for Caffe

        stage->attrs().set<float>("coordinates_offset", 1.0f);
        stage->attrs().set<bool>("initial_clip", false);
        stage->attrs().set<bool>("shift_anchors", false);
        stage->attrs().set<bool>("round_ratios", true);
        stage->attrs().set<bool>("swap_xy", false);
    }

    auto scales = layer->GetParamAsFloats("scale", {});
    auto ratios = layer->GetParamAsFloats("ratio", {});

    stage->attrs().set("scales", scales);
    stage->attrs().set("ratios", ratios);

    int number_of_anchors = ratios.size() * scales.size();

    // Allocate slightly larger buffer than needed for handling remnant in distribution among SHAVEs
    int buffer_size = (inputs[0]->desc().dim(Dim::H) + 16) * inputs[0]->desc().dim(Dim::W) * number_of_anchors * 5 * sizeof(float);

    model->addTempBuffer(
        stage,
        DataDesc({buffer_size}));
}

}  // namespace vpu
