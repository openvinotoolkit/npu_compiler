//
// INTEL CONFIDENTIAL
// Copyright 2019 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//

#include <vpu/pass_manager.hpp>

#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <cmath>
#include <list>
#include <set>
#include <unordered_map>
#include <memory>

#include <vpu/stub_stage.hpp>
#include <vpu/sw/utility.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

using ReplicatedDataMap = std::unordered_map<int, Data>;

class UpsamplingStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<UpsamplingStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>&,
            ScalePropagationStep) override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        _orderInfo.setInput(_inputEdges[0], DimsOrder::NCHW);
        _orderInfo.setOutput(_outputEdges[0], DimsOrder::NCHW);
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        _stridesInfo.setOutput(_outputEdges[0], StridesRequirement().add(1, DimStride::Aligned));
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        _batchInfo.setInput(_inputEdges[0], BatchSupport::Split);
        _batchInfo.setOutput(_outputEdges[0], BatchSupport::Split);
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::TwoOrOne;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto scaleX = attrs().get<int>("upsampling_factorx_x");
        auto scaleY = attrs().get<int>("upsampling_factorx_y");
        auto scaleZ = attrs().get<int>("upsampling_factorx_z");
        auto pad_l_x = attrs().get<int>("pad_l_x");
        auto pad_r_x = attrs().get<int>("pad_r_x");
        auto pad_l_y = attrs().get<int>("pad_l_y");
        auto pad_r_y = attrs().get<int>("pad_r_y");
        auto pad_l_z = attrs().get<int>("pad_l_z");
        auto pad_r_z = attrs().get<int>("pad_r_z");

        serializer.append(static_cast<int32_t>(scaleX));
        serializer.append(static_cast<int32_t>(scaleY));
        serializer.append(static_cast<int32_t>(scaleZ));
        serializer.append(static_cast<int32_t>(pad_l_x));
        serializer.append(static_cast<int32_t>(pad_r_x));
        serializer.append(static_cast<int32_t>(pad_l_y));
        serializer.append(static_cast<int32_t>(pad_r_y));
        serializer.append(static_cast<int32_t>(pad_l_z));
        serializer.append(static_cast<int32_t>(pad_r_z));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        input->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};


class DeconvolutionToConvolutionContent final : public CalculatedDataContent {
public:
    DeconvolutionToConvolutionContent(
            const DataContent::Ptr& origContent,
            int kernelSizeX, int kernelSizeY) :
            CalculatedDataContent({origContent}),
            _kerneSizeX(kernelSizeX), _kernelSizeY(kernelSizeY) {
    }

    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const {
        VPU_PROFILE(DeconvolutionToConvolutionContent);

        IE_ASSERT(baseContents.size() == 1);
        IE_ASSERT(_desc.type() == DataType::FP16);

        deconv_to_conv(baseContents[0]->get<fp16_t>(), static_cast<fp16_t*>(tempBuf), _desc);
    }

private:
    int _kerneSizeX;
    int _kernelSizeY;
};


class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(replaceDeconvByConv);

    auto stages = model->getStages();
    for (const auto& stage : stages) {
        if (stage->type() != StageType::StubDeconv) {
            continue;
        }

        auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        auto kernelStrideX = stage->attrs().get<int>("kernelStrideX");
        auto kernelStrideY = stage->attrs().get<int>("kernelStrideY");
        auto groupSize = stage->attrs().get<int>("groupSize");

        auto padLeft  = stage->attrs().get<int>("padLeft");
        auto padRight = stage->attrs().get<int>("padRight");
        auto padTop = stage->attrs().get<int>("padTop");
        auto padBottom = stage->attrs().get<int>("padBottom");
        auto deconvScale = stage->attrs().getOrDefault<float>("scaleFactor", 1.0);

        /* Upsampling layer does not support negative paddings */
        if ((kernelSizeX - 1 - padLeft < 0) || (kernelSizeX - 1 - padRight < 0) ||
            (kernelSizeY - 1 - padTop < 0) || (kernelSizeY - 1 - padBottom < 0)) {
            continue;
        }

        if (groupSize != 1) {
            continue;
        }

        if ((padTop != padBottom) || (padLeft != padRight)) {
            continue;
        }

        if (kernelSizeX > 15 || kernelSizeY > 15) {
            continue;
        }

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases  = stage->input(2);
        auto output = stage->output(0);
        const auto& env = CompileEnv::get();

        if (env.netConfig.hwDisabled(stage->origLayer()->name)) {
            continue;
        }

        if (output->desc().numDims() < 4) {
            continue;
        }

        // problem with Deconv/CommonSingleLayerTest
        auto origOutputX = kernelStrideX * (input->desc().dim(Dim::W)  - 1) + kernelSizeX - padLeft - padRight;
        auto origOutputY = kernelStrideY * (input->desc().dim(Dim::H)  - 1) + kernelSizeY - padTop - padBottom;

        if ((origOutputX != output->desc().dim(Dim::W)) || (origOutputY != output->desc().dim(Dim::H))) {
            continue;
        }

        model->disconnectStageDatas(stage);

        DataDesc newDesc({1, 1, output->desc().dim(Dim::C), output->desc().dim(Dim::N)});
        newDesc.setDim(Dim::N, input->desc().dim(Dim::N));
        newDesc.setDim(Dim::C, input->desc().dim(Dim::C));
        newDesc.setDim(Dim::H, (input->desc().dim(Dim::H) - 1) * kernelStrideY + 1 + (kernelSizeY - 1) * 2 - padTop - padBottom);
        newDesc.setDim(Dim::W, (input->desc().dim(Dim::W) - 1) * kernelStrideX + 1 + (kernelSizeX - 1) * 2 - padLeft - padRight);

        auto newOutput = model->duplicateData(output, "@upsampleData", newDesc);
        auto newWeights = model->duplicateData(weights, "@upsampleData", weights->desc(),
                     std::make_shared<DeconvolutionToConvolutionContent>(weights->content(), kernelSizeX, kernelSizeY));

        auto upsampleStage = model->addNewStage<UpsamplingStage>(
                stage->origLayerName() + "@Upsample",
                StageType::Upsampling,
                stage->origLayer(),
                {input},
                {newOutput});

        upsampleStage->attrs().set<int>("upsampling_factorx_x", kernelStrideX);
        upsampleStage->attrs().set<int>("upsampling_factorx_y", kernelStrideY);
        upsampleStage->attrs().set<int>("upsampling_factorx_z", 1);
        upsampleStage->attrs().set<int>("pad_l_x", (kernelSizeX - 1) - padLeft);
        upsampleStage->attrs().set<int>("pad_r_x", (kernelSizeX - 1) - padRight);
        upsampleStage->attrs().set<int>("pad_l_y", (kernelSizeY - 1) - padTop);
        upsampleStage->attrs().set<int>("pad_r_y", (kernelSizeY - 1) - padBottom);
        upsampleStage->attrs().set<int>("pad_l_z", 0);
        upsampleStage->attrs().set<int>("pad_r_z", 0);

        auto newStage = model->addNewStage<StubStage>(
                stage->origLayerName() + "@UpsampleConv",
                StageType::StubConv,
                stage->origLayer(),
                {newOutput, newWeights, biases},
                {output});

        newStage->attrs().set<int>("kernelSizeX", kernelSizeX);
        newStage->attrs().set<int>("kernelSizeY", kernelSizeY);
        newStage->attrs().set<int>("kernelStrideX", 1);
        newStage->attrs().set<int>("kernelStrideY", 1);
        newStage->attrs().set<int>("padLeft", 0);
        newStage->attrs().set<int>("padRight", 0);
        newStage->attrs().set<int>("padTop", 0);
        newStage->attrs().set<int>("padBottom", 0);
        newStage->attrs().set<int>("dilationX", 1);
        newStage->attrs().set<int>("dilationY", 1);
        newStage->attrs().set<int>("groupSize", 1);
        newStage->attrs().set<bool>("tryHW", true);
        newStage->attrs().set<float>("scaleFactor", deconvScale);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::replaceDeconvByConv() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
