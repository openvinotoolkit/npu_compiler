//
// Copyright 2018-2019 Intel Corporation.
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

#include <vpu/pass_manager.hpp>

#include <cmath>

#include <tuple>
#include <list>
#include <string>
#include <limits>
#include <algorithm>
#include <utility>
#include <vector>
#include <memory>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/stub_stage.hpp>
#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/tiling.hpp>
#include <vpu/hw/utility.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuidler) : _stageBuilder(stageBuidler) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

bool supportedPaddingPool(const Stage& stage) {
    IE_ASSERT(StageType::StubMaxPool == stage->type() ||
              StageType::StubAvgPool == stage->type());

    auto input  = stage->input(0);
    auto output = stage->output(0);

    auto kernelSizeX  = stage->attrs().get<int>("kernelSizeX");
    auto kernelSizeY  = stage->attrs().get<int>("kernelSizeY");
    auto kernelStride = stage->attrs().get<int>("kernelStrideX");
    auto padLeft      = stage->attrs().get<int>("padLeft");
    auto padRight     = stage->attrs().get<int>("padRight");
    auto padTop       = stage->attrs().get<int>("padTop");
    auto padBottom    = stage->attrs().get<int>("padBottom");

    //
    // Even kernel size with odd input -> HW bug
    // Need to add extra border
    //

    bool forcePaddingStage = false;

    if (kernelSizeX % 2 == 0 && input->desc().dim(Dim::W) % 2 == 1) {
        if (padRight == 0) {
            stage->attrs().set<int>("padRight", 1);
        }

        forcePaddingStage = true;
    }

    if (kernelSizeY % 2 == 0 && input->desc().dim(Dim::H) % 2 == 1) {
        if (padBottom == 0) {
            stage->attrs().set<int>("padBottom", 1);
        }

        forcePaddingStage = true;
    }

    auto hwInitialPad = getHwPaddingInfo(
        input->desc().dims(), output->desc().dims(),
        kernelSizeX, kernelSizeY,
        kernelStride, kernelStride);

    bool originalUnsupportedPad = (
        (padRight  != padLeft && padRight  != padLeft + 1)       ||
        (padBottom != padTop  && padBottom != padTop + 1)        ||
        (padLeft   != 0       && padLeft   != (kernelSizeX / 2)) ||
        (padRight  != 0       && padRight  != (kernelSizeX / 2)) ||
        (padTop    != 0       && padTop    != (kernelSizeY / 2)) ||
        (padBottom != 0       && padBottom != (kernelSizeY / 2)));

    bool hwUnsupportedPad = (
        (hwInitialPad.right  != hwInitialPad.left && hwInitialPad.right  != hwInitialPad.left + 1) ||
        (hwInitialPad.bottom != hwInitialPad.top  && hwInitialPad.bottom != hwInitialPad.top + 1)  ||
        (hwInitialPad.left   != 0                 && hwInitialPad.left   != (kernelSizeX / 2))     ||
        (hwInitialPad.right  != 0                 && hwInitialPad.right  != (kernelSizeX / 2))     ||
        (hwInitialPad.top    != 0                 && hwInitialPad.top    != (kernelSizeY / 2))     ||
        (hwInitialPad.bottom != 0                 && hwInitialPad.bottom != (kernelSizeY / 2)));

    return !originalUnsupportedPad &&
           !hwUnsupportedPad       &&
           !forcePaddingStage;
}

bool supportedPaddingConv(const Stage& stage) {
    IE_ASSERT(StageType::StubConv == stage->type());

    auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
    auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
    auto padLeft     = stage->attrs().get<int>("padLeft");
    auto padRight    = stage->attrs().get<int>("padRight");
    auto padTop      = stage->attrs().get<int>("padTop");
    auto padBottom   = stage->attrs().get<int>("padBottom");

    return (padRight  == padLeft) &&
           (padBottom == padTop)  &&
           (padLeft   == 0 || padLeft == (kernelSizeX / 2)) &&
           (padTop    == 0 || padTop  == (kernelSizeY / 2));
}

void insertPaddingStageBefore(const Model::Ptr& model, StageBuilder::Ptr& stageBuilder, const Stage& origStage) {
    auto origInput       = origStage->input(0);
    auto paddedInputDesc = origInput->desc();

    auto padLeft   = origStage->attrs().get<int>("padLeft");
    auto padRight  = origStage->attrs().get<int>("padRight");
    auto padTop    = origStage->attrs().get<int>("padTop");
    auto padBottom = origStage->attrs().get<int>("padBottom");

    paddedInputDesc.setDim(Dim::W, origInput->desc().dim(Dim::W) + padLeft + padRight);
    paddedInputDesc.setDim(Dim::H, origInput->desc().dim(Dim::H) + padTop + padBottom);

    auto inputPadded = model->duplicateData(
        origInput,
        "@padded",
        paddedInputDesc);

    model->replaceStageInput(origStage->inputEdge(0), inputPadded);

    auto paddingStage = stageBuilder->addPadStage(
        model,
        origStage->name() + "@padding",
        origStage->origLayer(),
        (origStage->type() == StageType::StubMaxPool) ? PadMode::Edge : PadMode::Constant,
        0.0f,
        DimValues({
            {Dim::W, padLeft},
            {Dim::H, padTop},
        }),
        DimValues({
            {Dim::W, padRight},
            {Dim::H, padBottom},
        }),
        origInput,
        inputPadded);

    origStage->attrs().set<int>("padLeft",   0);
    origStage->attrs().set<int>("padRight",  0);
    origStage->attrs().set<int>("padTop",    0);
    origStage->attrs().set<int>("padBottom", 0);
}

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(hwPadding);

    auto isPooling = [](const Stage& stage) {
        return StageType::StubMaxPool == stage->type() ||
               StageType::StubAvgPool == stage->type();
    };
    auto isConv = [](const Stage& stage) {
        return StageType::StubConv == stage->type();
    };

    auto stages = model->getStages();

    for (const auto& origStage : stages) {
        if (!isPooling(origStage) && !isConv(origStage)) {
            continue;
        }

        auto tryHW = origStage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        bool addPaddingStage = false;

        if (isConv(origStage)) {
            addPaddingStage = !supportedPaddingConv(origStage);
        } else if (isPooling(origStage)) {
            addPaddingStage = !supportedPaddingPool(origStage);
        } else {
            IE_ASSERT(false);
        }

        if (addPaddingStage) {
            insertPaddingStageBefore(model, _stageBuilder, origStage);
        }
    }
}

}  // namespace

Pass::Ptr PassManager::hwPadding() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
