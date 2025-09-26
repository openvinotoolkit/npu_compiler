//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/adaptive_stripping_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/asymmetric_quant_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/compressed_convolution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/profiling_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/qdq_optimization_aggressive_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sep_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/static_shape_op_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_table_reuse_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SETTARGETINDEPENDENTPASSOPTIONS
#define GEN_PASS_DEF_SETTARGETINDEPENDENTPASSOPTIONS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

using vpux::VPU::getAttributeFromOption;

//
// SetTargetIndependentPassOptionsPass
//

class SetTargetIndependentPassOptionsPass final :
        public VPU::impl::SetTargetIndependentPassOptionsBase<SetTargetIndependentPassOptionsPass> {
public:
    SetTargetIndependentPassOptionsPass() = default;
    SetTargetIndependentPassOptionsPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(initCompilerOptions);
    }
    mlir::LogicalResult initialize(mlir::MLIRContext* context) override final;

private:
    void safeRunOnModule() override final;

    bool _allowCustomValues = false;
    llvm::SmallVector<std::pair<llvm::StringRef, mlir::Attribute>, /* expected num Opts*/ 14> _optionSet;
};

mlir::LogicalResult SetTargetIndependentPassOptionsPass::initialize(mlir::MLIRContext* context) {
    _optionSet = {
            {VPU::AUTO_PADDING_ODU, getAttributeFromOption(context, enableAutoPaddingODU)},
            {VPU::AUTO_PADDING_IDU, getAttributeFromOption(context, enableAutoPaddingIDU)},
            {VPU::ASYMMETRIC_PER_TENSOR_ZP, getAttributeFromOption(context, enableAsymmetricPerTensorZP)},
            {VPU::ASYMMETRIC_PER_CHANNEL_ZP, getAttributeFromOption(context, enableAsymmetricPerChannelZP)},
            {VPU::REDUCE_SUPPORTED, getAttributeFromOption(context, enableIsReduceSupported)},
            {VPU::FP16_COMPRESSED_CONV, getAttributeFromOption(context, enableFP16CompressedConvolution)},
            {VPU::VPUNN_PRE_SPLIT, getAttributeFromOption(context, enableVPUNNPreSplit)},
            {VPU::ENABLE_SE_PTRS_OPERATIONS, getAttributeFromOption(context, enableSEPtrsOperations)},
            {VPU::ENABLE_EXPERIMENTAL_SE_PTRS_OPERATIONS,
             getAttributeFromOption(context, enableExperimentalSEPtrsOperations)},
            {VPU::ENABLE_ADAPTIVE_STRIPPING, mlir::BoolAttr::get(context, enableQDQOptimizationAggressive.getValue() ||
                                                                                  enableAdaptiveStripping.getValue())},
            {VPU::ENABLE_QDQ_OPTIMIZATION_AGGRESSIVE, getAttributeFromOption(context, enableQDQOptimizationAggressive)},
            {VPU::ENABLE_EXTRA_STATIC_SHAPE_OPS, getAttributeFromOption(context, enableExtraStaticShapeOps)},
            {VPU::WEIGHTS_TABLE_REUSE_MODE, getAttributeFromOption(context, weightsTableReuseMode)},
            {VPU::ENABLE_PROFILING, getAttributeFromOption(context, enableProfiling)},
    };

    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }
    return mlir::success();
}

void SetTargetIndependentPassOptionsPass::safeRunOnModule() {
    auto moduleOp = getModuleOp(getOperation());
    auto optionsBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    auto pipelineOptionsOp = VPU::getPipelineOptionsOp(getContext(), moduleOp);
    optionsBuilder =
            mlir::OpBuilder::atBlockBegin(&pipelineOptionsOp.getOptions().front(), optionsBuilder.getListener());

    auto* ctx = optionsBuilder.getContext();
    for (const auto& [name, attribute] : _optionSet) {
        bool hasPipelineOption = pipelineOptionsOp.lookupSymbol<config::OptionOp>(name) != nullptr;
        VPUX_THROW_WHEN(!_allowCustomValues && hasPipelineOption,
                        "Option {0} is already defined, probably you run '--init-compiler' twice", name);

        if (hasPipelineOption) {
            continue;
        }
        optionsBuilder.create<config::OptionOp>(optionsBuilder.getUnknownLoc(), mlir::StringAttr::get(ctx, name),
                                                attribute);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createSetTargetIndependentPassOptionsPass() {
    return std::make_unique<SetTargetIndependentPassOptionsPass>();
}
std::unique_ptr<mlir::Pass> vpux::VPU::createSetTargetIndependentPassOptionsPass(
        const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    return std::make_unique<SetTargetIndependentPassOptionsPass>(initCompilerOptions, log);
}
