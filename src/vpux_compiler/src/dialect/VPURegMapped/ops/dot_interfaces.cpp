#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

using namespace vpux;

namespace {
void printIndex(llvm::raw_ostream& os, VPURegMapped::IndexType index, llvm::StringRef head, llvm::StringRef middle,
                llvm::StringRef end) {
    os << head << "Index: " << middle << index.getTileIdx() << ":" << index.getListIdx() << ":" << index.getValue()
       << end;
}

}  // namespace

//
// FetchTask
//

DotNodeColor VPURegMapped::FetchTaskOp::getNodeColor() {
    return DotNodeColor::BLUE;
}

bool VPURegMapped::FetchTaskOp::printAttributes(llvm::raw_ostream& os, llvm::StringRef head, llvm::StringRef middle,
                                                llvm::StringRef end) {
    printIndex(os, getType(), head, middle, end);
    return true;
}

DOT::EdgeDir VPURegMapped::FetchTaskOp::getEdgeDirection(mlir::Operation* source) {
    if (source->getNumResults() > 0) {
        auto it = llvm::find_if(source->getResults(), [this](mlir::Value val) {
            return this->getPreviousTask() == val;
        });
        if (it != source->getResults().end()) {
            return DOT::EdgeDir::EDGE_NORMAL;
        };
    }
    return DOT::EdgeDir::EDGE_REVERSE;
}

//
// EnqueueOp
//

DotNodeColor VPURegMapped::EnqueueOp::getNodeColor() {
    return DotNodeColor::GREEN;
}

bool VPURegMapped::EnqueueOp::printAttributes(llvm::raw_ostream& os, llvm::StringRef head, llvm::StringRef middle,
                                              llvm::StringRef end) {
    printIndex(os, getType(), head, middle, end);
    return true;
}

DOT::EdgeDir VPURegMapped::EnqueueOp::getEdgeDirection(mlir::Operation* source) {
    auto res = source->getResult(0);

    if (getStart() == res || getStart() == res) {
        return DOT::EdgeDir::EDGE_REVERSE;
    }

    return DOT::EdgeDir::EDGE_NORMAL;
}
