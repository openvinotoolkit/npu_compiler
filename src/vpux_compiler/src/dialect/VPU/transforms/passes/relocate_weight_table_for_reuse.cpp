//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_RELOCATEWEIGHTTABLEFORREUSE
#define GEN_PASS_DEF_RELOCATEWEIGHTTABLEFORREUSE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class RelocateWeightTableForReusePass final :
        public VPU::impl::RelocateWeightTableForReuseBase<RelocateWeightTableForReusePass> {
public:
    explicit RelocateWeightTableForReusePass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void RelocateWeightTableForReusePass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPU::NCEConvolutionOp op) {
        _log.trace("[{0}]: NCE ConvolutionOp at {1}", op->getName(), op->getLoc());
        Const::DeclareOp cstOp = nullptr;
        auto unrolledOp = mlir::dyn_cast<VPU::UnrolledTypeOp>(op.getWeightsTable().getDefiningOp());
        if (unrolledOp != nullptr) {
            cstOp = mlir::dyn_cast<Const::DeclareOp>(unrolledOp.getInput().getDefiningOp());
        } else {
            cstOp = mlir::dyn_cast<Const::DeclareOp>(op.getWeightsTable().getDefiningOp());
        }

        if (cstOp == nullptr) {
            return;
        }

        const bool isDistrType = unrolledOp != nullptr &&
                                 mlir::dyn_cast<VPU::DistributedTensorType>(op.getWeightsTable().getType()) != nullptr;
        if (op.getFilter().getType().isa<VPU::SparseTensorType>()) {
            return;
        }

        const auto channelOffset = 0;
        auto weightTableType = op.getWeightsTable().getType().cast<vpux::NDTypeInterface>();
        auto shapeTotalSize = weightTableType.getShape().totalSize();
        auto elementSize = weightTableType.getElemTypeSize().count() / CHAR_BIT;
        int64_t weightsElemBitSize = CHAR_BIT;
        if (auto weights = op.getFilter()) {
            weightsElemBitSize = getElemTypeSize(weights.getType()).count();
        }

        size_t numClusters = 1;
        SmallVector<int64_t> offsets;
        if (isDistrType) {
            auto distrType = op.getWeightsTable().getType().cast<VPU::DistributedTensorType>();
            numClusters = distrType.getDistribution().getNumClusters().getInt();
            const auto perClusterShapeOffsets = distrType.getPerClusterMemoryShapeOffsets();
            VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                              "Mismatch between the number of shape offsets '{0}' and the number of clusters '{1}'.",
                              perClusterShapeOffsets.size(), numClusters);
            for (auto clusterOffsets : perClusterShapeOffsets | indexed) {
                offsets.push_back(clusterOffsets.value()[Dims4D::Filter::OC]);
            }
        } else {
            offsets.push_back(0);
        }

        SmallVector<uint32_t> weightsPtrPerCluster(numClusters, static_cast<int32_t>(0));
        auto originalOC = 0;
        if (VPU::canAutopadOutput(op.getOperation())) {
            originalOC = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getShape()[Dims4D::Act::C];
        }
        auto newConstAttr = cstOp.getContentAttr()
                                    .transform()
                                    .relocateWeightsTablePointers(
                                            weightsPtrPerCluster, VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY,
                                            ShapeRef(offsets), (shapeTotalSize * elementSize), weightsElemBitSize,
                                            nullptr, channelOffset, originalOC)
                                    .get();

        mlir::OpBuilder builder(cstOp);
        auto newConstOp =
                builder.create<Const::DeclareOp>(cstOp.getLoc(), cstOp.getOutput().getType(), std::move(newConstAttr));
        vpux::Const::foldSingleConstant(newConstOp);

        if (isDistrType) {
            unrolledOp->setOperand(0, newConstOp.getOutput());
        } else {
            op.setOperand(2, newConstOp.getOutput());
        }
        if (cstOp->getUses().empty()) {
            cstOp.erase();
        }
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createRelocateWeightTableForReusePass(Logger log) {
    return std::make_unique<RelocateWeightTableForReusePass>(std::move(log));
}
