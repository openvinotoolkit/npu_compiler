//
// INTEL CONFIDENTIAL
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <algorithm>
#include <set>

#include <vpu/compile_env.hpp>

namespace vpu {

void FrontEnd::parseInputAndOutputData(const Model::Ptr& model) {
    VPU_PROFILE(parseInputAndOutputData);

    const auto& env = CompileEnv::get();

    //
    // Parse network inputs
    //

    for (const auto& inputInfo : _ieNetworkParser.networkInputs) {
        auto netInput = inputInfo.second;
        IE_ASSERT(netInput != nullptr);

        auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        DataDesc vpuDesc(ieData->getTensorDesc());
        if (vpuDesc.numDims() >= 3) {
            if (env.config.hwOptimization || env.config.forceLayout == ComputeLayout::NCHW) {
                vpuDesc.moveDim(Dim::C, 2);
            } else {
                vpuDesc.moveDim(Dim::C, 0);
            }
        }

        auto vpuData = model->addInputData(ieData->getName(), vpuDesc);
        bindData(vpuData, ieData);
    }

    model->attrs().set<int>("numInputs", _ieNetworkParser.networkInputs.size());

    //
    // Parse network outputs
    //

    for (const auto& outputInfo : _ieNetworkParser.networkOutputs) {
        auto ieData = outputInfo.second;
        IE_ASSERT(ieData != nullptr);

        DataDesc vpuDesc(ieData->getTensorDesc());
        if (vpuDesc.numDims() >= 3) {
            if (env.config.hwOptimization || env.config.forceLayout == ComputeLayout::NCHW) {
                vpuDesc.moveDim(Dim::C, 2);
            } else {
                vpuDesc.moveDim(Dim::C, 0);
            }
        }

        auto vpuData = model->addOutputData(ieData->getName(), vpuDesc);
        bindData(vpuData, ieData);

        if (_unbatchedOutputs.count(ieData) > 0) {
            vpuData->attrs().set<bool>("unbatched", true);
        }
    }

    model->attrs().set<int>("numOutputs", _ieNetworkParser.networkOutputs.size());

    //
    // Parse constant data
    //

    for (const auto& constInfo : _ieNetworkParser.constDatas) {
        auto ieData = constInfo.first;
        IE_ASSERT(ieData != nullptr);

        auto ieBlob = constInfo.second;
        IE_ASSERT(ieBlob != nullptr);

        auto ieDesc = ieData->getTensorDesc();

        if (ieDesc.getPrecision() != ie::Precision::FP16) {
            if (ieDesc.getPrecision() != ie::Precision::FP32 || !env.config.allowFP32Models) {
                VPU_THROW_EXCEPTION << "Unsupported precision " << ieDesc.getPrecision() << "for data " << ieData->getName();
            }
        }

        DataDesc vpuDesc(ieDesc);
        vpuDesc.setType(DataType::FP16);

        auto vpuData = model->addConstData(
            ieData->getName(),
            vpuDesc,
            ieBlobContent(ieBlob));

        // User might ask to return the output from Const layer.
        if (auto vpuOutData = getVpuData(ieData)) {
            IE_ASSERT(vpuOutData->usage() == DataUsage::Output);

            _stageBuilder->addCopyStage(
                model,
                formatString("%s@return-const", vpuData->name()),
                nullptr,
                vpuData,
                vpuOutData);
        }

        bindData(vpuData, ieData);
    }

    //
    // Add Copy stages after network outputs, if they are in the middle
    //

    for (const auto& outputInfo : _ieNetworkParser.networkOutputs) {
        auto ieData = outputInfo.second;
        IE_ASSERT(ieData != nullptr);

        auto vpuData = getVpuData(ieData);
        IE_ASSERT(vpuData != nullptr);

        // It might be Const.
        if (vpuData->usage() != DataUsage::Output)
            continue;

        // Convert stage will be added.
        if (vpuData->desc().type() != DataType::FP16)
            continue;

        if (!ieData->getInputTo().empty()) {
            auto vpuTempData = model->duplicateData(
                vpuData,
                "@intermediate",
                vpuData->desc());

            _stageBuilder->addCopyStage(
                model,
                formatString("%s@copy-to-output", vpuData->name()),
                nullptr,
                vpuTempData,
                vpuData);

            bindData(vpuTempData, ieData);
        }
    }
}

}  // namespace vpu
