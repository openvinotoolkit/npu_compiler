//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_COMPUTESESIZES
#define GEN_PASS_DEF_COMPUTESESIZES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

SmallVector<VPUIP::NCEClusterTaskOp> findProducerOps(mlir::Value value) {
    SmallVector<VPUIP::NCEClusterTaskOp> producerNCEOps;

    auto producerOp = value.getDefiningOp();

    if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(producerOp)) {
        producerNCEOps.push_back(nceOp);
    } else if (mlir::isa<VPUIP::CopyOp, VPUIP::NNDMAOp>(producerOp)) {
        const auto ops = findProducerOps(producerOp->getOperand(0));
        producerNCEOps.append(ops);
    } else if (VPUIP::isPureViewOp(producerOp)) {
        llvm::TypeSwitch<mlir::Operation*, void>(producerOp)
                .Case<VPUIP::ConcatViewOp>([&](VPUIP::ConcatViewOp concatOp) {
                    for (auto input : concatOp.getInputs()) {
                        const auto ops = findProducerOps(input);
                        producerNCEOps.append(ops);
                    }
                })
                .Case<mlir::ViewLikeOpInterface>([&](mlir::ViewLikeOpInterface viewOp) {
                    const auto ops = findProducerOps(viewOp.getViewSource());
                    producerNCEOps.append(ops);
                })
                .Case<MultiViewOpInterface>([&](MultiViewOpInterface viewOp) {
                    if (auto opResult = value.dyn_cast<mlir::OpResult>()) {
                        const auto source = viewOp.getViewSource(opResult.getResultNumber());
                        const auto ops = findProducerOps(source);
                        producerNCEOps.append(ops);
                    }
                })
                .Case<GroupedViewOpInterface>([&](GroupedViewOpInterface viewOp) {
                    for (auto source : viewOp.getViewSources()) {
                        const auto ops = findProducerOps(source);
                        producerNCEOps.append(ops);
                    }
                });
    }

    return producerNCEOps;
}

// Get the number of channels produced by the operation's variants. Except of the last one, Each variant must produce
// the same number of channels in order for the sparse output to be valid.
int64_t getVariantsNumChannels(VPUIP::NCEClusterTaskOp nceOp) {
    std::optional<int64_t> numChannels = std::nullopt;
    auto variants = to_small_vector(nceOp.getVariants().getOps<VPUIP::DPUTaskOp>());
    if (variants.size() > 1) {
        variants.pop_back();
    }

    for (auto dpuTaskOp : variants) {
        const auto outStart = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutStart());
        const auto outEnd = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutEnd());
        VPUX_THROW_UNLESS(outStart.size() >= 3 && outEnd.size() >= 3,
                          "Invalid variant shape, expected at least three dimensions, got '{0}'", outStart.size());
        const auto variantNumChannels = outEnd[2] - outStart[2] + 1;
        if (!numChannels.has_value()) {
            numChannels = variantNumChannels;
            continue;
        }
        VPUX_THROW_UNLESS(numChannels.value() == variantNumChannels,
                          "Variant has '{0}' channels while '{1}' were expected", variantNumChannels, numChannels);
    }
    return numChannels.value();
}

std::optional<int64_t> getInputSESizeForConcatOverC(mlir::Value operand) {
    const auto producerOps = findProducerOps(operand);
    VPUX_THROW_WHEN(producerOps.empty(), "No producer operations found");

    const auto producerOpsNumChannels = to_small_vector(producerOps | transformed([](VPUIP::NCEClusterTaskOp nceOp) {
                                                            return getVariantsNumChannels(nceOp);
                                                        }));
    const auto unequalChannels = std::adjacent_find(producerOpsNumChannels.begin(), producerOpsNumChannels.end(),
                                                    std::not_equal_to<>()) != producerOpsNumChannels.end();
    VPUX_THROW_WHEN(unequalChannels, "Not all producer output channels are equal");

    // Check if the number of output channels of the producer variants is different than the consumer operation's
    // input channels, as this represents a case where the activations are concatenated over channels
    const auto producerChannels = producerOpsNumChannels.front();
    const auto consumerChannels = operand.getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
    if (producerChannels != consumerChannels) {
        return producerChannels;
    }
    return std::nullopt;
}

//
// ComputeSESizesPass
//

class ComputeSESizesPass final : public VPUIP::impl::ComputeSESizesBase<ComputeSESizesPass> {
public:
    explicit ComputeSESizesPass(std::optional<bool> onlyInputsConcatOverC, Logger log)
            : _onlyInputsConcatOverC(onlyInputsConcatOverC) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    std::optional<bool> _onlyInputsConcatOverC;
};

mlir::LogicalResult ComputeSESizesPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!onlyInputsConcatOverC.hasValue()) {
        return mlir::success();
    }
    _onlyInputsConcatOverC = onlyInputsConcatOverC;
    return mlir::success();
}

void ComputeSESizesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto arch = VPU::getArch(func);
    auto constraint = VPU::getSparsityConstraint(arch);

    // Set the storage element size attributes only for the input operand in case the sparse data is concatenated
    // over channels. This is necessary since sparse activations are sparsified individually by each DPU producer
    // and must be desparsified in the same way.
    // Consumers of sparse activations have a single storage element size register configuration (i.e. se_z_split).
    // Therefore, sparse activations concatenated over channels must all have the same number of channels,
    // regardless of whether the concatenation is done explicitly, using ODU broadcasting, by multiple variants
    // of the same invariant, or a combination of these methods.
    if (_onlyInputsConcatOverC.value_or(false)) {
        _log.trace("Setting storage element sizes only for inputs concatenated over channels");

        func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
            if (nceOp.getInputStorageElementTable() != nullptr) {
                _log.trace("Skipping operation '{0}' at '{1}' which has a storage element table", nceOp->getName(),
                           nceOp->getLoc());
                return;
            }

            _log.trace("Handling operation '{0}' at '{1}'", nceOp->getName(), nceOp->getLoc());

            if (nceOp.getInputSparsityMap() != nullptr) {
                auto producerChannels = getInputSESizeForConcatOverC(nceOp.getInput());
                if (producerChannels.has_value()) {
                    VPUX_THROW_UNLESS(
                            constraint.areChannelsFitForSESize(nceOp.getInput().getType(), producerChannels.value()),
                            "Invalid number of channels '{0}' for concatenated input", producerChannels.value());
                    auto seSizeAttr = getIntAttr(&ctx, producerChannels.value());
                    _log.nest().trace("Setting input_se_size to '{0}'", seSizeAttr);
                    nceOp.setInputSeSizeAttr(seSizeAttr);
                }
            }

            if (nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE && nceOp.getWeightsSparsityMap() != nullptr) {
                auto producerChannels = getInputSESizeForConcatOverC(nceOp.getWeights());
                if (producerChannels.has_value()) {
                    const auto inputSeSize = nceOp.getInputSeSize().value_or(producerChannels.value());
                    VPUX_THROW_UNLESS(
                            inputSeSize == producerChannels.value(),
                            "Different storage element sizes expected for the two Eltwise inputs: {0} and {1} at '{2}'",
                            inputSeSize, producerChannels.value(), nceOp->getLoc());
                }
            }
        });

        return;
    }

    // Set the storage element sizes to be equal to the number of channels of the activation, if not already set.
    _log.trace("Setting storage element sizes for all sparse activations");

    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        _log.trace("Handling operation '{0}' at '{1}'", nceOp->getName(), nceOp->getLoc());

        if (nceOp.getInputSeSizeAttr() == nullptr) {
            if (nceOp.getInputStorageElementTable() != nullptr) {
                VPUX_THROW_WHEN(nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE,
                                "Explicit SETable for Eltwise operations is not yet supported");

                auto inputType = nceOp.getInput().getType().cast<vpux::NDTypeInterface>();
                auto seTableType = nceOp.getInputStorageElementTable().getType().cast<vpux::NDTypeInterface>();
                const auto numChannels = inputType.getShape()[Dims4D::Act::C];
                const auto seDepth = seTableType.getShape()[Dims4D::Act::C];
                VPUX_THROW_WHEN(numChannels % seDepth != 0, "Storage element size is not an integer");
                const auto seSize = numChannels / seDepth;
                VPUX_THROW_UNLESS(constraint.areChannelsFitForSESize(seSize),
                                  "Invalid storage element size '{0}' for input", seSize);

                _log.nest().trace("Setting input_se_size to '{0}' [SE]", seSize);
                nceOp.setInputSeSizeAttr(getIntAttr(&ctx, seSize));
            } else if (nceOp.getInputSparsityMap() != nullptr) {
                auto inputType = nceOp.getInput().getType().cast<vpux::NDTypeInterface>();
                auto numChannels = inputType.getShape()[Dims4D::Act::C];
                VPUX_THROW_UNLESS(constraint.areChannelsFitForSESize(inputType, numChannels),
                                  "Invalid number of channels '{0}' for input", numChannels);
                _log.nest().trace("Setting input_se_size to '{0}' [SM]", numChannels);
                nceOp.setInputSeSizeAttr(getIntAttr(&ctx, numChannels));

                if (nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE) {
                    auto numChannelsInput2 =
                            nceOp.getWeights().getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
                    VPUX_THROW_UNLESS(
                            numChannels == numChannelsInput2,
                            "Different storage element sizes expected for the two Eltwise inputs: {0} and {1}",
                            numChannels, numChannelsInput2);
                }
            }
        }

        if (nceOp.getOutputSparsityMapBuff() != nullptr && nceOp.getOutputSeSizeAttr() == nullptr) {
            auto numChannels = getVariantsNumChannels(nceOp);
            VPUX_THROW_UNLESS(constraint.areChannelsFitForSESize(numChannels),
                              "Invalid number of channels '{0}' for output", numChannels);
            _log.nest().trace("Setting output_se_size to '{0}'", numChannels);
            nceOp.setOutputSeSizeAttr(getIntAttr(&ctx, numChannels));
        }
    });
}

}  // namespace

//
// createComputeSESizesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createComputeSESizesPass(std::optional<bool> onlyInputsConcatOverC,
                                                                  Logger log) {
    return std::make_unique<ComputeSESizesPass>(onlyInputsConcatOverC, log);
}
