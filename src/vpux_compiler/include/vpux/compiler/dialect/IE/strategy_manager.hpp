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

#pragma once

#include <map>
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/utils/core/checked_cast.hpp"
namespace vpux {

constexpr llvm::StringLiteral multiClusterStrategyAttrName = "multiClusterStrategy";

class OperationEfficiencyTable {
    using tableMap = std::map<int, std::map<int, int>>;
    tableMap operationEfficiencyTable;

    class proxy {
        const std::map<int, int>& mMap;

    public:
        proxy(const std::map<int, int>& m): mMap(m) {
        }

        int operator[](int x) const {
            std::map<int, int>::const_iterator iter = mMap.find(x);
            if (iter == mMap.end()) {
                VPUX_THROW("Unable to find value '{0}' in oepration efficency table", x);
            } else {
                return iter->second;
            }
        }
    };

public:
    OperationEfficiencyTable() {
    }
    OperationEfficiencyTable(const std::map<int, std::map<int, int>>& o): operationEfficiencyTable(o) {
    }

    proxy operator[](int x) const {
        std::map<int, std::map<int, int>>::const_iterator iter = operationEfficiencyTable.find(x);
        if (iter == operationEfficiencyTable.end()) {
            VPUX_THROW("Unable to find value '{0}' in oepration efficency table", x);
        } else {
            return proxy(iter->second);
        }
    }
};

//
// StrategyManager
//

class StrategyManager final {
public:
    explicit StrategyManager(mlir::FuncOp func, size_t numClusters, Logger log);

public:
    void computeOptimalMultiClusterStrategy();

private:
    template <class ConcreteOp>
    bool isOperationSplitOverHeightCompatible(ConcreteOp op);
    template <class ConcreteOp>
    bool isOperationSplitOverKernelCompatible(ConcreteOp op);
    size_t calculateSplitOverHeightEfficency(mlir::Operation* op);
    size_t calculateSplitOverKernelEfficency(mlir::Operation* op);
    void assignMultiClusterStrategy(mlir::Operation* op);
    std::map<int64_t, std::map<int64_t, double>> channelMajorEfficiencyTable();
    std::map<int64_t, std::map<int64_t, double>> depthwiseEfficiencyTable();

    const size_t _minimumHeightForSOH = 20;
    const size_t _minimumOutputChannelsPerCluster = 16;
    llvm::DenseMap<mlir::Operation*, size_t> _splitOverHeightEfficencies;
    llvm::DenseMap<mlir::Operation*, size_t> _splitOverKernelEfficencies;
    size_t _numClusters;
    Logger _log;
    mlir::FuncOp _func;
};

template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverHeightCompatible(ConcreteOp op) {
    const auto outputShape = getShape(op.output());
    const auto OH = outputShape[Dims4D::Act::H];
    return OH >= _minimumHeightForSOH;
}

template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverKernelCompatible(ConcreteOp op) {
    const auto outputShape = getShape(op.output());
    const auto OC = outputShape[Dims4D::Act::C];
    return OC >= _minimumOutputChannelsPerCluster * _numClusters;
}

}  // namespace vpux
