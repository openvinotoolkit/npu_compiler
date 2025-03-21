//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/sparsity_strategy.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/Support/ThreadPool.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_SPARSIFYWEIGHTS
#define GEN_PASS_DEF_SPARSIFYWEIGHTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// SparsifyWeightsPass
//

class SparsifyWeightsPass final : public VPU::impl::SparsifyWeightsBase<SparsifyWeightsPass> {
public:
    explicit SparsifyWeightsPass(VPU::WeightsSparsityHeuristic heuristic, std::optional<double> manualThreshold,
                                 int64_t largeConstThreshold, int64_t computeOpThreshold, bool enableWeightSwizzling,
                                 Logger log)
            : _heuristic(heuristic),
              _manualThreshold(manualThreshold),
              _largeConstThreshold(checkThreshold(largeConstThreshold)),
              _computeOpThreshold(computeOpThreshold),
              _enableWeightSwizzling(enableWeightSwizzling) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    static Byte checkThreshold(int64_t threshold) {
        VPUX_THROW_UNLESS(threshold >= 0, "Weights sparsity largeConstThreshold is negative '{0}'", threshold);
        return Byte(threshold);
    }

private:
    VPU::WeightsSparsityHeuristic _heuristic;
    std::optional<double> _manualThreshold;
    const Byte _largeConstThreshold;
    const Byte _tinyConstThreshold = Byte(1024);
    const int64_t _computeOpThreshold = 350;
    const bool _enableWeightSwizzling = true;
};

//
// safeRunOnFunc
//

int64_t getSizeInfo(Const::DeclareOp& origOp) {
    const auto& contentAttr = origOp.getContentAttr();
    auto inputType = contentAttr.getBaseContent().getType().cast<NDTypeInterface>();
    auto maximumSize = inputType.getTotalAllocSize().count();
    auto transformations = contentAttr.getTransformations();
    for (auto transformation : transformations) {
        inputType = transformation.inferOutputType(inputType);
        auto size = inputType.getTotalAllocSize().count();
        maximumSize = (size > maximumSize) ? size : maximumSize;
    }
    return maximumSize;
}

int64_t getSizeInfo(mlir::Value input, mlir::Value output) {
    const auto inSize = mlir::cast<NDTypeInterface>(input.getType()).getTotalAllocSize().count();
    const auto outSize = mlir::cast<NDTypeInterface>(output.getType()).getTotalAllocSize().count();
    return inSize + outSize;
}

void SparsifyWeightsPass::safeRunOnFunc() {
    using namespace VPU::NCESparsity;

    auto func = getOperation();
    auto& ctx = getContext();

    // fragmentation prevention config
    int64_t numTiles = 0;
    int64_t minWeightsSize = 0;
    int64_t smallOpThreshold = 0;

    const auto computeOpCount = static_cast<int64_t>(to_small_vector(func.getOps<VPU::ClusteredOpInterface>()).size());
    // fragmentation likely only with longer execution series
    // experimental number for minimum number of compute operations where fragmentation is likely
    const auto fragmentationPossible = computeOpCount > _computeOpThreshold && _enableWeightSwizzling;

    if (fragmentationPossible) {
        // update config for fragmentation
        auto module = getModuleOp(func);
        auto tileOp = IE::getTileExecutor(module);
        numTiles = tileOp.getCount();

        // for cases with weight swizzling enabled the address is aligned to a value
        // with weights sparsity both weights and sparsity map will need to be aligned
        // increasing fragmentation likelihood with small aligned constants
        // avoid increasing size more than 4X
        minWeightsSize = getAddressAlignmentForSwizzling(vpux::SWIZZLING_KEY_5, VPU::getArch(func)) / 4;

        // experimental number for small ops which do not suffer from fragmentation
        smallOpThreshold = 2560;
    }

    std::unique_ptr<BaseWeightsSparsityStrategy> enablementStrategy;
    if (_heuristic == VPU::WeightsSparsityHeuristic::CMX) {
        _log.trace("Using CMX-based heuristic");
        const Byte availableCMX = VPU::getTotalCMXSize(func);
        enablementStrategy = std::make_unique<CMXConsumptionBasedWeightsSparsityStrategy>(
                availableCMX, CMX_BASED_STRATEGY_DEFAULT_INTERVALS, _manualThreshold);
    } else if (_heuristic == VPU::WeightsSparsityHeuristic::RATIO) {
        _log.trace("Using ratio-based heuristic");
        enablementStrategy = std::make_unique<RatioBasedWeightsSparsityStrategy>(
                WEIGHTS_SPARSITY_FLOAT_RATIO_THRESHOLD, WEIGHTS_SPARSITY_INT_RATIO_THRESHOLD, _manualThreshold);
    } else {
        VPUX_THROW("Unsupported heuristic: {0}", _heuristic);
    }

    int64_t numCandidatesSparseWeights = 0;
    int64_t numSparsifiedWeights = 0;

    DenseMap<Const::DeclareOp, SmallVector<VPU::SparseOpInterface>> sparseCandidates;

    auto innerLog = _log.nest();

    // Walk the IR and find all sparse weights candidates
    func->walk([&](VPU::SparseOpInterface sparsifiableOp) {
        if (!VPU::supportsSparseWeights(sparsifiableOp.getOperation())) {
            return;
        }

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(sparsifiableOp.getOperation());
        VPUX_THROW_UNLESS(nceOp != nullptr, "Unexpected non-NCE operation that supports weights sparsity");

        const auto weights = nceOp.getWeightsOperand();
        if (weights == nullptr || mlir::isa<mlir::BlockArgument>(weights)) {
            return;
        }
        auto weightsType = weights.getType().cast<vpux::NDTypeInterface>();
        if (weightsType.getElemTypeSize().count() < CHAR_BIT) {
            _log.trace("Op '{0}' at '{1}' is not supporting sparsity for sub 8-bit weights", sparsifiableOp->getName(),
                       sparsifiableOp->getLoc());
            return;
        }

        _log.trace("Op '{0}' at '{1}' is a candidate for sparsifying its weights", sparsifiableOp->getName(),
                   sparsifiableOp->getLoc());

        if (weights.getType().isa<VPU::SparseTensorType>()) {
            innerLog.trace("Weights are already sparse");
            return;
        }

        auto weightsOp = weights.getDefiningOp<Const::DeclareOp>();
        if (weightsOp == nullptr) {
            innerLog.trace("Expected weights parent to be constant, but got '{0}'", weights.getDefiningOp()->getName());
            return;
        }

        if (fragmentationPossible && numTiles > 0) {
            const auto weightsOpSize = getSizeInfo(weightsOp);
            const auto activationSize = getSizeInfo(nceOp->getOperand(0), nceOp->getResult(0));
            if (weightsOpSize > smallOpThreshold && activationSize > smallOpThreshold &&
                weightsOpSize > activationSize) {
                // weight bound operation above "small" threshold
                if (weightsOpSize / numTiles <= minWeightsSize) {
                    // possible fragmentation with swizzling
                    innerLog.trace("Weight bound compute op '{0}' at '{1}' does not satisfy minWeightsSize",
                                   sparsifiableOp->getName(), sparsifiableOp->getLoc());
                    return;
                }
            }
        }

        if (mlir::isa<vpux::VPU::NCECompressConvolutionOp>(sparsifiableOp)) {
            innerLog.trace("Operation uses the compressed convolution feature. Skipping");
            return;
        }

        sparseCandidates[weightsOp].push_back(sparsifiableOp);
        ++numCandidatesSparseWeights;
    });

    DenseMap<Const::DeclareOp, VPU::GroupSparseTensorOp> localReplacementCache;
    // poor man's way to only create a GroupSparseTensor once. done this way to
    // limit the amount of changes around multi-threaded code.
    const auto getCachedSparseTensorOp = [&](Const::DeclareOp weightsOp, mlir::DenseElementsAttr data,
                                             const Const::ContentSetup& newContentAttrSetup,
                                             VPU::SparsityCompressionAttr sparsityCompressionAttr) {
        auto it = localReplacementCache.find(weightsOp);
        if (it != localReplacementCache.end()) {
            return it->second;
        }

        mlir::OpBuilder builder(weightsOp);
        auto sparsityMapContent = Const::ContentAttr::get(data, newContentAttrSetup.clone().getSparsityMap());
        auto sparsifiedContent = Const::ContentAttr::get(data, newContentAttrSetup.clone().sparsify(false));
        const auto sparsifiedWeights = builder.create<Const::DeclareOp>(weightsOp.getLoc(), sparsifiedContent.getType(),
                                                                        std::move(sparsifiedContent));
        const auto sparsityMap = builder.create<Const::DeclareOp>(weightsOp.getLoc(), sparsityMapContent.getType(),
                                                                  std::move(sparsityMapContent));
        auto groupedView =
                builder.create<VPU::GroupSparseTensorOp>(weightsOp.getLoc(), sparsifiedWeights->getResult(0),
                                                         sparsityMap->getResult(0), true, sparsityCompressionAttr);

        it = localReplacementCache.insert({weightsOp, groupedView}).first;
        return it->second;
    };

    std::mutex irModificationMutex;

    // In parallel, count sparse elements in sparse candidates and decide which ones should be made sparse
    const auto tryToSparsify = [&](VPU::SparseOpInterface sparsifiableOp, Const::DeclareOp weightsOp,
                                   const Const::Content& foldedContent) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(sparsifiableOp.getOperation());
        const auto weights = nceOp.getWeightsOperand();
        auto weightsType = weights.getType().cast<vpux::NDTypeInterface>();

        const auto foldedElemType = foldedContent.getType().getElementType();
        const auto inputType = sparsifiableOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto hasFloatInput = inputType.getElementType().isa<mlir::FloatType>();
        const auto numNonSparseElemsPerOC = vpux::countNonSparseElementsPerOC(foldedContent, foldedElemType);
        if (!enablementStrategy->shouldSparsifyWeights(innerLog, weightsType, numNonSparseElemsPerOC, hasFloatInput)) {
            innerLog.trace("Weights will not be sparsified", sparsifiableOp->getName(), sparsifiableOp->getLoc());
            return;
        }
        innerLog.trace("Sparsifying weights for op '{0}' at '{1}'", sparsifiableOp->getName(),
                       sparsifiableOp->getLoc());
        const auto numElemsType =
                mlir::RankedTensorType::get({static_cast<int64_t>(numNonSparseElemsPerOC.size())}, getInt64Type(&ctx));
        const auto numElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(numNonSparseElemsPerOC));
        const auto axisAttr = getIntAttr(&ctx, Dims4D::Filter::OC.ind());
        const auto alignmentAttr = getIntAttr(&ctx, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
        const auto sparsityCompressionAttr =
                VPU::SparsityCompressionAttr::get(&ctx, axisAttr, numElemsAttr, alignmentAttr);

        // Fold the original constant to drop the original transformations
        // This is done in order to avoid repeating the folding that was done in this pass later in the compilation
        auto foldedContentType = foldedContent.getType();
        if (auto qType = foldedElemType.dyn_cast<mlir::quant::QuantizedType>()) {
            foldedContentType = foldedContentType.changeElemType(normalizeQuantStorageType(qType));
        }
        // It is necessary to copy the contents of the folded constant into a new buffer since the data conversion
        // does not take place until the values are extracted (e.g. inside `Const::Content::copyTo`)
        // For example, if the original constants are FP32, they will still occupy storage for FP32 elements after
        // folding until they are stored in a new buffer allocated for INT8 data and copied into it
        const auto contentSize = checked_cast<size_t>(foldedContentType.getTotalAllocSize().count());
        std::vector<char> newContent(contentSize);
        foldedContent.copyTo(MutableArrayRef(newContent.data(), contentSize));
        const auto foldedBaseContent =
                Const::createConstContent(mlir::cast<mlir::ShapedType>(foldedContentType), ArrayRef(newContent));
        Const::ContentSetup newContentAttrSetup(foldedContentType);
        // Folded constants with INT8 element types have to be cast to quantized types for the correct type to be
        // inferred from the new Const::ContentAttr
        if (auto qType = foldedElemType.dyn_cast<mlir::quant::QuantizedType>()) {
            newContentAttrSetup = newContentAttrSetup.castElemType(qType);
        }

        // IR modification is not thread safe according to MLIR documentation.
        std::lock_guard<std::mutex> guard(irModificationMutex);

        VPU::GroupSparseTensorOp groupedView =
                getCachedSparseTensorOp(weightsOp, foldedBaseContent, newContentAttrSetup, sparsityCompressionAttr);

        weightsOp->replaceUsesWithIf(groupedView, [useToReplace = sparsifiableOp.getOperation()](mlir::OpOperand& use) {
            return use.getOwner() == useToReplace;
        });
        if (weightsOp->getUses().empty()) {
            weightsOp->erase();
        }

        ++numSparsifiedWeights;
    };

    // If multi-threading is not enabled, sparsify all weights sequentially
    if (!ctx.isMultithreadingEnabled()) {
        for (auto& [weightsOp, sparsifiableOps] : sparseCandidates) {
            for (auto sparsifiableOp : sparsifiableOps) {
                tryToSparsify(sparsifiableOp, weightsOp, weightsOp.getContent());
            }
        }
    } else {
        SmallVector<Const::DeclareOp> parallelSet;
        DenseMap<Const::DeclareOp, int64_t> constSizes;

        // Sparsify all constants that are within tinyConstThreshold or exceed largeConstThreshold
        // by using a single thread.
        for (auto& [weightsOp, sparsifiableOps] : sparseCandidates) {
            int64_t constSize = getSizeInfo(weightsOp);
            constSizes[weightsOp] = constSize;
            if (constSize > _tinyConstThreshold.count() && constSize < _largeConstThreshold.count()) {
                parallelSet.push_back(weightsOp);
                continue;
            }
            for (auto sparsifiableOp : sparsifiableOps) {
                tryToSparsify(sparsifiableOp, weightsOp, weightsOp.getContent());
            }
        }

        // Iterate over other constant operations and accumulate constants until the combined size
        // exceeds the large constant threshold or the end is reached.
        auto& threadPool = ctx.getThreadPool();
        auto op = parallelSet.begin();
        auto opsEnd = parallelSet.end();

        while (op != opsEnd) {
            int64_t constSizeInParallel = 0;
            unsigned int constNumInParallel = 0;
            auto opReserve = op;
            while (constSizeInParallel <= _largeConstThreshold.count() && op != opsEnd) {
                auto weightsOpInParallel = *op;
                int64_t cstSize = constSizes[weightsOpInParallel];
                constSizeInParallel += cstSize;
                constNumInParallel++;
                ++op;
            }

            mlir::ParallelDiagnosticHandler handler(&ctx);
            std::atomic<unsigned int> curIndex(0);
            auto processConstants = [&] {
                while (true) {
                    const unsigned int index = curIndex++;
                    if (index >= constNumInParallel) {
                        break;
                    }
                    handler.setOrderIDForThread(index);
                    Const::DeclareOp curOp = *(std::next(opReserve, index));
                    for (auto sparsifiableOp : sparseCandidates[curOp]) {
                        tryToSparsify(sparsifiableOp, curOp, curOp.getContent());
                    }
                    handler.eraseOrderIDForThread();
                }
            };

            llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
            unsigned int numActions = std::min(constNumInParallel, threadPool.getThreadCount());
            for (unsigned int i = 0; i < numActions; ++i) {
                tasksGroup.async(processConstants);
            }
            tasksGroup.wait();
        }
    }

    _log.trace("Sparsified weights for {0} operations out of {1} candidates", numSparsifiedWeights,
               numCandidatesSparseWeights);
}

}  // namespace

//
// createSparsifyWeightsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSparsifyWeightsPass(VPU::WeightsSparsityHeuristic heuristic,
                                                                 std::optional<double> manualThreshold,
                                                                 int64_t largeConstThreshold,
                                                                 int64_t computeOpThreshold, bool enableWeightSwizzling,
                                                                 Logger log) {
    return std::make_unique<SparsifyWeightsPass>(heuristic, manualThreshold, largeConstThreshold, computeOpThreshold,
                                                 enableWeightSwizzling, log);
}
