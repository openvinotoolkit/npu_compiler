//
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <memory>
#include <string>
#include <set>
#include <vector>

#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

class ConvertStage final : public StageNode {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<ConvertStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto inputScale = inputScales[0];

        if (_type == StageType::Convert_f16f32) {
            IE_ASSERT(output->usage() == DataUsage::Output);
            IE_ASSERT(step == ScalePropagationStep::Propagate);

            _scaleInfo.setInput(_inputEdges[0], 1.0f);
            _scaleInfo.setOutput(_outputEdges[0], 1.0f);
        } else {
            IE_ASSERT(input->usage() == DataUsage::Input);

            _scaleInfo.setOutput(_outputEdges[0], inputScale);

            if (step == ScalePropagationStep::ScaleInput) {
                attrs().get<float>("scale") *= inputScale;
                attrs().get<float>("bias") *= inputScale;
            }
        }
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        if (_type == StageType::Convert_f16f32) {
            IE_ASSERT(output->usage() == DataUsage::Output);

            auto outDimsOrder = output->desc().dimsOrder();

            // HCW is not supported
            IE_ASSERT(outDimsOrder.dimInd(Dim::C) != 1);

            _orderInfo.setInput(_inputEdges[0], outDimsOrder);
        } else {
            IE_ASSERT(input->usage() == DataUsage::Input);

            auto inDimsOrder = input->desc().dimsOrder();

            // HCW is not supported
            IE_ASSERT(inDimsOrder.dimInd(Dim::C) != 1);

            _orderInfo.setOutput(_outputEdges[0], inDimsOrder);
        }
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto inDimsOrder = input->desc().dimsOrder();

        StridesRequirement reqs;

        if (input->desc().dim(Dim::N, 1) > 1) {
            // To merge batch into previous dimension.
            reqs.add(inDimsOrder.dimInd(Dim::N), DimStride::Compact);
        }

        if (_type == StageType::Convert_f16f32) {
            IE_ASSERT(output->usage() == DataUsage::Output);

            _stridesInfo.setInput(_inputEdges[0], reqs);
            _stridesInfo.setOutput(_outputEdges[0], StridesRequirement::compact());
        } else {
            IE_ASSERT(input->usage() == DataUsage::Input);

            _stridesInfo.setInput(_inputEdges[0], StridesRequirement::compact());
            _stridesInfo.setOutput(_outputEdges[0], reqs);
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
        // Convert will support batch by merging it with previous dimension.
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        // TODO: more SHAVEs leads to hang on public MTCNN network with U8 input
        return StageSHAVEsRequirements::TwoOrOne;
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto scale = attrs().get<float>("scale");
        auto bias = attrs().get<float>("bias");
        auto convertFromDetOutput = attrs().getOrDefault<bool>("convertFromDetOutput", false);
        auto haveBatch = attrs().getOrDefault<bool>("haveBatch", true);

        serializer.append(static_cast<float>(scale));
        serializer.append(static_cast<float>(bias));
        serializer.append(static_cast<int32_t>(convertFromDetOutput));
        serializer.append(static_cast<int32_t>(haveBatch));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        if (input->desc().dimsOrder() == DimsOrder::NC) {
            input->serializeOldBuffer(
                handle_from_this(),
                serializer,
                DimsOrder::HWC,
                {
                    {Dim::W, {Dim::N}},
                    {Dim::C, {Dim::C}}
                });

            output->serializeOldBuffer(
                handle_from_this(),
                serializer,
                DimsOrder::HWC,
                {
                    {Dim::W, {Dim::N}},
                    {Dim::C, {Dim::C}}
                });
        } else if (input->desc().dim(Dim::N, 1) > 1) {
            auto perm = input->desc().dimsOrder().toPermutation();
            IE_ASSERT(perm.size() == 4);

            auto batchDimInd = input->desc().dimsOrder().dimInd(Dim::N);
            IE_ASSERT(batchDimInd == perm.size() - 1);

            input->serializeOldBuffer(
                handle_from_this(),
                serializer,
                DimsOrder::HWC,
                {
                    {Dim::H, {perm[2], perm[3]}},
                    {Dim::W, {perm[1]}},
                    {Dim::C, {perm[0]}}
                });

            output->serializeOldBuffer(
                handle_from_this(),
                serializer,
                DimsOrder::HWC,
                {
                    {Dim::H, {perm[2], perm[3]}},
                    {Dim::W, {perm[1]}},
                    {Dim::C, {perm[0]}}
                });
        } else {
            input->serializeOldBuffer(handle_from_this(), serializer);

            output->serializeOldBuffer(handle_from_this(), serializer);
        }
    }
};

}  // namespace

Stage StageBuilder::createConvertStage(
        const Model::Ptr& model,
        const std::string& name,
        const Data& input,
        const Data& output,
        StageType type,
        float scale,
        float bias) {
    auto stage = model->addNewStage<ConvertStage>(
        name,
        type,
        nullptr,
        {input},
        {output});

    stage->attrs().set("scale", scale);
    stage->attrs().set("bias", bias);

    return stage;
}

void FrontEnd::addDataTypeConvertStages(const Model::Ptr& model) {
    VPU_PROFILE(addDataTypeConvertStages);

    const auto& env = CompileEnv::get();

    if (env.config.inputScale != 1.f) {
        env.log->warning("[VPU] GraphTransformer : INPUT_NORM option is deprecated");
    }

    if (env.config.inputBias != 0.f) {
        env.log->warning("[VPU] GraphTransformer : INPUT_BIAS option is deprecated");
    }

    for (const auto& input : model->datas()) {
        if (input->usage() != DataUsage::Input)
            continue;

        if (input->desc().type() != DataType::FP16) {
            env.log->debug("convert input %s to FP16", input->name());

            auto fp16Desc = input->desc();
            fp16Desc.setType(DataType::FP16);

            auto inputFP16 = model->duplicateData(
                input,
                "@FP16",
                fp16Desc);

            input->attrs().set<Data>("fp16_copy", inputFP16);

            bindData(inputFP16, input->origData());

            auto stageType = StageType::None;
            switch (input->desc().type()) {
            case DataType::U8:
                stageType = StageType::Convert_u8f16;
                break;
            case DataType::FP32:
                stageType = StageType::Convert_f32f16;
                break;
            default:
                VPU_THROW_EXCEPTION << "Unsupported input data type : " << input->desc().type();
            }

            _stageBuilder->createConvertStage(
                model,
                inputFP16->name(),
                input,
                inputFP16,
                stageType,
                env.config.inputScale,
                env.config.inputBias);
        } else if (env.config.inputScale != 1.0f || env.config.inputBias != 0.0f) {
            std::ostringstream postfixOstr;
            if (env.config.inputScale != 1.0f) {
                postfixOstr << "@SCALE=" << std::to_string(env.config.inputScale);
            }
            if (env.config.inputBias != 0.0f) {
                postfixOstr << "@BIAS=" << std::to_string(env.config.inputBias);
            }

            auto postfix = postfixOstr.str();

            auto scaledInput = model->duplicateData(
                input,
                postfix);

            bindData(scaledInput, input->origData());

            _stageBuilder->addPowerStage(
                model,
                scaledInput->name(),
                nullptr,
                env.config.inputScale,
                1.0f,
                env.config.inputBias,
                input,
                scaledInput);
        }
    }

    for (const auto& output : model->datas()) {
        if (output->usage() != DataUsage::Output)
            continue;

        if (output->desc().type() != DataType::FP16) {
            env.log->debug("convert output %s from FP16", output->name());

            IE_ASSERT(output->desc().type() == DataType::FP32);

            auto fp16Desc = output->desc();
            fp16Desc.setType(DataType::FP16);

            auto outputFP16 = model->duplicateData(
                output,
                "@FP16",
                fp16Desc);

            output->attrs().set<Data>("fp16_copy", outputFP16);

            bindData(outputFP16, output->origData());

            auto stage = _stageBuilder->createConvertStage(
                model,
                outputFP16->name(),
                outputFP16,
                output,
                StageType::Convert_f16f32);

            auto withDetectionOutput = model->attrs().getOrDefault<bool>("withDetectionOutput", false);
            stage->attrs().set<bool>("convertFromDetOutput", withDetectionOutput);

            auto haveBatch = _unbatchedOutputs.count(output->origData()) == 0;
            stage->attrs().set<bool>("haveBatch", haveBatch);
        }
    }
}

}  // namespace vpu
