//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <numeric>

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>

//
// DefinedInArch
//
namespace vpux {
namespace config {
template <config::ArchKind arch>
struct DefinedInArch {
    template <typename ConcreteType>
    class Impl : public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
    public:
        static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
            return verifyArchKind(op, arch);
        }

    private:
        static mlir::LogicalResult verifyArchKind(mlir::Operation* op, config::ArchKind definedInArch) {
            auto actualArch = config::getArch(op);

            if (actualArch != config::ArchKind::UNKNOWN && actualArch < definedInArch) {
                auto actualArchStr = stringifyArchKind(actualArch).str();
                auto definedInArchStr = stringifyArchKind(definedInArch).str();
                return vpux::errorAt(op, "Operation {0} not supported in {1}; op has been introduced in {2}",
                                     op->getName(), actualArchStr, definedInArchStr);
            }

            return mlir::success();
        }
    };
};

//
// LimitedToArch
//

template <config::ArchKind... archs>
struct LimitedToArch {
    template <typename ConcreteType>
    class Impl : public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
    public:
        static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
            return verifyArchKind(op, {archs...});
        }

    private:
        static mlir::LogicalResult verifyArchKind(mlir::Operation* op,
                                                  std::initializer_list<config::ArchKind> supportedArchs) {
            auto actualArch = config::getArch(op);

            if (actualArch != config::ArchKind::UNKNOWN) {
                if (std::find(cbegin(supportedArchs), cend(supportedArchs), actualArch) == cend(supportedArchs)) {
                    auto actualArchStr = stringifyArchKind(actualArch).str();
                    auto archsStr = std::accumulate(
                            cbegin(supportedArchs), cend(supportedArchs), std::string(),
                            [](const std::string& accu, const config::ArchKind arch) -> std::string {
                                return accu + (accu.length() > 0 ? "," : "") + stringifyArchKind(arch).str();
                            });
                    return vpux::errorAt(op, "Operation {0} not supported in {1}; list of supported archs: {2}",
                                         op->getName(), actualArchStr, archsStr);
                }
            }

            return mlir::success();
        }
    };
};
}  // namespace config
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/config/ops_interfaces.hpp.inc>
