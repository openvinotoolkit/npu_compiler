//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;
namespace {

//
// OptimizeParallelCopiesPass
//

class OptimizeParallelCopiesPass final : public IERT::OptimizeParallelCopiesBase<OptimizeParallelCopiesPass> {
public:
    explicit OptimizeParallelCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

bool isCopyFusable(IERT::CopyOp copyOp) {
    // Check 1: copy DDR->DMA
    const auto srcMemory = VPU::getMemoryKind(copyOp.input().getType().cast<mlir::MemRefType>());
    const auto dstMemory = VPU::getMemoryKind(copyOp.output().getType().cast<mlir::MemRefType>());
    if (srcMemory == dstMemory || srcMemory == VPU::MemoryKind::CMX_NN) {
        //        std::cout << "\tfail at 1" << std::endl;
        return false;
    }

    // Check 2: parallel
    // All the consumer of the parent op should be copies
    // At least one more copy except for the current one
    auto parentOp = copyOp.input().getDefiningOp();
    if (parentOp == nullptr) {
        return false;
    }
    if (mlir::isa<Const::DeclareOp>(parentOp)) {
        //        std::cout<<llvm::formatv("FORBIDDEN WEIGHTS SHARING {0}, {1}", parentOp->getLoc(),
        //        parentOp->getName()).str()<<std::endl;
        return false;
    }
    bool hasSubView = false;
    auto subViewOp = mlir::dyn_cast<IERT::SubViewOp>(copyOp.input().getDefiningOp());
    if (mlir::isa<IERT::SubViewOp>(parentOp)) {
        hasSubView = true;
        std::cout << llvm::formatv("\t!!parentOp is subView {0}, {1}", parentOp->getLoc(), parentOp->getName()).str()
                  << std::endl;
        parentOp = mlir::dyn_cast<IERT::SubViewOp>(parentOp).source().getDefiningOp();
    }
    bool hasSiblingCopy = false;
    //    std::cout << llvm::formatv("\tparent op is {0}, {1}", parentOp->getLoc(), parentOp->getName()).str() <<
    //    std::endl;
    for (auto siblingOp : parentOp->getResult(0).getUsers()) {
        if (!mlir::isa<IERT::CopyOp>(*siblingOp)) {
            std::cout << llvm::formatv("\tparentOp at {0}, {1}", parentOp->getLoc(), parentOp->getName()).str()
                      << std::endl;
            std::cout
                    << llvm::formatv("\tsibling not copy at {0}, {1}", siblingOp->getLoc(), siblingOp->getName()).str()
                    << std::endl;
            if (!mlir::isa<IERT::SubViewOp>(*siblingOp)) {
                continue;
            } else {
                auto childOfSiblingOp = *siblingOp->getResult(0).getUsers().begin();
                std::cout << llvm::formatv("\tfound subview, check its child at {0}, {1}", childOfSiblingOp->getLoc(),
                                           childOfSiblingOp->getName())
                                     .str()
                          << std::endl;
                if (!mlir::isa<IERT::CopyOp>(childOfSiblingOp)) {
                    continue;
                }
                // match SubView->Copy
                if (!hasSubView) {
                    continue;
                }
                auto siblingSubViewOp = mlir::dyn_cast<IERT::SubViewOp>(siblingOp);
                if (parseIntArrayAttr<int64_t>(subViewOp.static_offsets()) !=
                            parseIntArrayAttr<int64_t>(siblingSubViewOp.static_offsets()) ||
                    parseIntArrayAttr<int64_t>(subViewOp.static_sizes()) !=
                            parseIntArrayAttr<int64_t>(siblingSubViewOp.static_sizes())) {
                    continue;
                }
                siblingOp = childOfSiblingOp;
            }
        }

        // Check 3: current op's consumer is copied to DDR immediately after execution
        auto childOfSiblingOp = siblingOp->getResult(0).getUsers().begin();
        auto childCopyOfSiblingOp = mlir::dyn_cast<IERT::CopyOp>(*childOfSiblingOp->getResult(0).getUsers().begin());
        if (childCopyOfSiblingOp == nullptr) {
            return false;
        }
        if (VPU::getMemoryKind(childCopyOfSiblingOp.input().getType().cast<mlir::MemRefType>()) !=
                    VPU::MemoryKind::CMX_NN ||
            VPU::getMemoryKind(childCopyOfSiblingOp.output().getType().cast<mlir::MemRefType>()) !=
                    VPU::MemoryKind::DDR) {
            //            std::cout << "\tfail at 3-1" << std::endl;
            return false;
        }

        if (siblingOp != copyOp) {
            //            std::cout << llvm::formatv("\tsibling op found {0}, {1}", siblingOp->getLoc(),
            //            siblingOp->getName()).str()
            //                      << std::endl;
            hasSiblingCopy = true;
        }
    }
    if (!hasSiblingCopy) {
        //        std::cout << "\tfail at 2-2" << std::endl;
        return false;
    }

    return true;
}

mlir::LogicalResult fuseParallelCopyOp(IERT::CopyOp copyOp, Logger log) {
    auto parentOp = copyOp.input().getDefiningOp();
    if (parentOp == nullptr) {
        return mlir::failure();
    }

    if (mlir::isa<IERT::SubViewOp>(parentOp)) {
        std::cout << "<<<<>>>> is SubView!" << std::endl;
        auto subViewOp = mlir::dyn_cast<IERT::SubViewOp>(parentOp);
        parentOp = subViewOp.source().getDefiningOp();
        auto getSiblingSubViews = [&]() -> SmallVector<IERT::SubViewOp> {
            SmallVector<IERT::SubViewOp> res;
            for (auto siblingOp : parentOp->getResult(0).getUsers()) {
                if (mlir::isa<IERT::SubViewOp>(*siblingOp) && siblingOp != subViewOp) {
                    auto siblingSubViewOp = mlir::dyn_cast<IERT::SubViewOp>(*siblingOp);
                    if (siblingSubViewOp.static_offsets() == subViewOp.static_offsets() &&
                        siblingSubViewOp.static_sizes() == subViewOp.static_sizes()) {
                        std::cout << llvm::formatv("<<<<>>>> push back {0} {1}", siblingSubViewOp->getLoc(),
                                                   siblingSubViewOp->getName())
                                             .str()
                                  << std::endl;
                        res.push_back(mlir::dyn_cast<IERT::SubViewOp>(*siblingOp));
                    }
                }
            }
            return res;
        };
        auto siblingSubViews = getSiblingSubViews();
        for (auto siblingSubView : siblingSubViews) {
            std::cout << llvm::formatv("\t<<<<>>>> merge {0}, {1}", siblingSubView->getLoc(), siblingSubView->getName())
                                 .str()
                      << std::endl;

            log.trace("Fuse SubView op {0} to {1}", subViewOp->getLoc(), siblingSubView->getLoc());
            auto siblingCopy = mlir::dyn_cast<IERT::CopyOp>(*siblingSubView.result().getUsers().begin());

            siblingSubView.getOperation()->replaceAllUsesWith(subViewOp.getOperation());
            siblingCopy.getOperation()->replaceAllUsesWith(copyOp);
            siblingCopy->erase();
            siblingSubView->erase();
        }

        // DEBUG LOG
        for (auto t : copyOp->getResult(0).getUsers()) {
            std::cout << llvm::formatv("\t\t~~~~ now the children: {0}, {1}", t->getLoc(), t->getName()).str()
                      << std::endl;
        }

        return mlir::success();
    }

    auto getSiblingCopies = [&]() -> SmallVector<IERT::CopyOp> {
        SmallVector<IERT::CopyOp> res;
        for (auto siblingOp : parentOp->getResult(0).getUsers()) {
            if (mlir::isa<IERT::CopyOp>(*siblingOp) && siblingOp != copyOp) {
                res.push_back(mlir::dyn_cast<IERT::CopyOp>(*siblingOp));
            }
        }
        return res;
    };
    auto siblingCopies = getSiblingCopies();
    for (auto siblingCopy : siblingCopies) {
        //        std::cout << llvm::formatv("\t~~~~ merge {0}, {1}", siblingCopy->getLoc(),
        //        siblingCopy->getName()).str()
        //                  << std::endl;

        log.trace("Fuse copy op {0} to {1}", copyOp->getLoc(), siblingCopy->getLoc());

        SmallVector<mlir::Operation*> wtUsingOutBuf;
        for (auto use : siblingCopy.output_buff().getUsers()) {
            //            std::cout<<llvm::formatv("!!!!! check users: {0} {1}", use->getLoc(),
            //            use->getName()).str()<<std::endl;
            if (mlir::isa<VPUIP::WeightsTableOp>(*use)) {
                wtUsingOutBuf.push_back(use);
            }
        }
        for (auto wt : wtUsingOutBuf) {
            wt->setOperand(0, copyOp.output_buff());
        }

        siblingCopy.getOperation()->replaceAllUsesWith(copyOp.getOperation());
        siblingCopy->erase();
    }

    // DEBUG LOG
    for (auto t : copyOp->getResult(0).getUsers()) {
        std::cout << llvm::formatv("\t\t~~~~ now the children: {0}, {1}", t->getLoc(), t->getName()).str() << std::endl;
    }

    return mlir::success();
}

// safeRunOnFunc

void OptimizeParallelCopiesPass::safeRunOnFunc() {
    getFunction()->walk([&](IERT::CopyOp copyOp) {
        if (isCopyFusable(copyOp)) {
            _log.nest(1).trace("Fuse parallel copy op '{0}' at '{1}'", copyOp->getName(), copyOp->getLoc());
            std::cout
                    << llvm::formatv("Fuse parallel copy op '{0}' at '{1}'", copyOp->getName(), copyOp->getLoc()).str()
                    << std::endl;
            if (mlir::failed(fuseParallelCopyOp(copyOp, _log))) {
                _log.nest(1).trace("Failed copy fusion of {0} at {1}", copyOp->getName(), copyOp->getLoc());
                std::cout
                        << llvm::formatv("Failed copy fusion of {0} at {1}", copyOp->getName(), copyOp->getLoc()).str()
                        << std::endl;
            }
        }
    });
}
}  // namespace

//
// createOptimizeParallelCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createOptimizeParallelCopiesPass(Logger log) {
    return std::make_unique<OptimizeParallelCopiesPass>(log);
}
