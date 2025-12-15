//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {

/*
   Interface for implementing platform specific rewriter patterns applied using the Greedy driver
*/
class IGreedilyPassStrategy {
public:
    virtual ~IGreedilyPassStrategy() = default;

    virtual void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const = 0;
};

/*
   Interface for implementing platform specific rewriter patterns applied using the Conversion driver
*/
class IConversionPassStrategy {
public:
    virtual ~IConversionPassStrategy() = default;

    virtual void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const = 0;
    virtual void markOpLegality(mlir::ConversionTarget& target, Logger& log) const = 0;
};

/*
   Interface for implementing platform specific rewriter patterns applied using the Iterative Walk Driver
*/
class IIterativeWalkPassStrategy {
public:
    virtual ~IIterativeWalkPassStrategy() = default;

    virtual void addPatterns(SmallVector<mlir::RewritePatternSet>& patterns, Logger& log) const = 0;
};

}  // namespace vpux
