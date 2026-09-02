//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"
#include "vpux/utils/ov/config.hpp"

#include <llvm/Support/Regex.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>

namespace vpux {

/**
 * @enum IRPrintingOrder
 * @brief VPUX IR pass printing before/after or before and after
 */
enum class IRPrintingOrder {
    BEFORE,
    AFTER,
    BEFORE_AFTER,
};

//
// DeveloperConfig
//

class DeveloperConfig final {
public:
    DeveloperConfig(Logger log, mlir::MLIRContext* ctx, const vpux::OV::Config& config);
    DeveloperConfig(const DeveloperConfig& other) = delete;
    DeveloperConfig& operator=(const DeveloperConfig& other) = delete;
    ~DeveloperConfig();

    void setup(mlir::DefaultTimingManager& tm) const;
    void setup(mlir::PassManager& pm, const vpux::OV::Config& config, bool isSubPipeline = false) const;
    void dump(mlir::PassManager& pm) const;

    bool useSharedConstants() const {
        return _useSharedConstants;
    }

private:
    Logger _log;

    std::string _crashReproducerFile;
    bool _localReproducer = true;

    std::string _irPrintingFilter;
    std::string _irPrintingLocation;
    std::string _irPrintingOrderStr;
    bool _printToFileTree = false;
    bool _printFullIR = false;
    bool _printFullConstant = false;
    bool _useSharedConstants = true;
    bool _allowPrintingHexConstant = true;
    bool _printDebugInfo = false;
    bool _printDebugInfoPrettyForm = false;
    bool _printSsaPrettyForm = true;
    std::string _printAsTextualPipelineFilePath = "";
    std::string _printDotOptions;
    std::string _dumpIntermediateValues;

    std::unique_ptr<llvm::Regex> _irDumpFilter;
    std::unique_ptr<llvm::raw_fd_ostream> _irDumpFile;
    llvm::raw_ostream* _irDumpStream = nullptr;
    IRPrintingOrder _irPrintingOrder = IRPrintingOrder::AFTER;
};

}  // namespace vpux
