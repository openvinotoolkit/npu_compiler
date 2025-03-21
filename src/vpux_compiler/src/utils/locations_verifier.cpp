//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/locations_verifier.hpp"

#include "vpux/compiler/core/developer_build_utils.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IERT/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"

#include "vpux/utils/core/type_traits.hpp"
#include "vpux/utils/profiling/common.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>

#include <unordered_map>

using namespace vpux;

namespace {
constexpr bool PRINT_NAME_DUPLICATES = true;
constexpr bool PRINT_OP_DUPLICATES = true;

constexpr StringRef LOCATIONS_VERIFICATION_MODE_ATTR_NAME = "IE.LocationsVerificationMode";

vpux::Logger getLogger() {
    return vpux::Logger("name-verifier", vpux::Logger::global().level());
}

template <typename First>
bool isOpFromAnyDialect(mlir::Operation* op) {
    return op->getDialect()->getNamespace() == First::getDialectNamespace();
}

template <typename First, typename Second, typename... Rest>
bool isOpFromAnyDialect(mlir::Operation* op) {
    return isOpFromAnyDialect<First>(op) || isOpFromAnyDialect<Second, Rest...>(op);
}

bool isOpFromIgnoredDialect(mlir::Operation* op) {
    // ELF dialects are ignored because locations are serialized before.
    return !isOpFromAnyDialect<Const::ConstDialect, IE::IEDialect, VPU::VPUDialect, IERT::IERTDialect,
                               VPUIP::VPUIPDialect, VPURT::VPURTDialect, mlir::BuiltinDialect, mlir::func::FuncDialect>(
            op);
}

bool isIgnoredOpType(mlir::Operation* op) {
    // Excluding IE/VPU/VPUIP casts ops, because they won't be translated to real SW/DMA tasks.DistributedCasts because
    // of CMXConcat dependency, track: E81319.
    const bool isViewLikeOp = IE::isPureViewOp(op) || VPU::isPureViewOp(op) || VPUIP::isPureViewOp(op);
    // This opperations needed to setup dependencies between blocks and not used anymore. Usually they just
    // duplicate location of inner op
    const bool isControlFlowOp = mlir::isa<mlir::async::YieldOp, mlir::async::ExecuteOp, mlir::async::AwaitOp,
                                           VPU::YieldOp, mlir::func::CallOp>(op);
    // Memory allocations doesn't represent real task, so no need to verify their uniqueness
    const bool isMemoryAllocOp =
            mlir::isa<VPURT::DeclareBufferOp, mlir::memref::AllocOp, VPUIP::StaticAllocOp, VPURT::Alloc>(op);
    // Similarly to memory allocations this operations aren't real tasks, they represent setup of hardware
    const bool isSetupOp = mlir::isa<VPURT::DeclareVirtualBarrierOp, VPURT::TaskOp, VPURT::ConfigureBarrierOp,
                                     VPUIP::PPETaskOp, Const::DeclareOp>(op);
    // Other locations. NCEClusterTilingOp is just wrapper, usually have same loc as inner. SparseBuffer[Un]Group is
    // just change of type representation, so no need to verify it
    bool isIgnoredOp = mlir::isa<VPUIP::NCEClusterTilingOp, VPUIP::GroupSparseBufferOp, VPUIP::UngroupSparseBufferOp,
                                 IE::DataInfoOp, mlir::UnrealizedConversionCastOp>(op);
    return isViewLikeOp || isControlFlowOp || isMemoryAllocOp || isSetupOp || isIgnoredOp;
}

bool hasExcludedPatterns(mlir::Operation* op) {
    const auto loc = op->getLoc();
    // Skip ReshapeOps introduced by type converter in ConvertShapeTo4D pass. This reshapes do nothing
    if (mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp>(op)) {
        const auto stringifiedLoc = stringifyPrimaryLocation(loc);
        return stringifiedLoc.find("materialized_") != std::string::npos;
    }
    // skip profiling related NameLocs
    if (mlir::isa<mlir::NameLoc>(loc)) {
        const auto stringifiedLoc = stringifyPrimaryLocation(loc);
        // Right now profiling LIT tests has locations in old format, so ignore them until all passes will be
        // handled
        const std::vector<std::string> profilingRelatedPatterns = {
                profiling::PROFILING_CMX_2_DDR_OP_NAME, profiling::PROFILING_DDR_2_DDR_OP_NAME,
                profiling::PROFILING_WORKPOINT_READ_ATTR, "combinedProfilingDataOutputInfo"};
        return std::any_of(profilingRelatedPatterns.begin(), profilingRelatedPatterns.end(),
                           [&](const std::string& pattern) {
                               return stringifiedLoc.find(pattern) != std::string::npos;
                           });
    }
    return false;
}

bool isIgnoredOp(mlir::Operation* op) {
    const bool hasUnknownLoc = op->getLoc().isa<mlir::UnknownLoc>();
    const bool isIgnored = hasUnknownLoc || isIgnoredOpType(op) || hasExcludedPatterns(op);
    return isIgnored;
}

// MLIR/LLVM types use own methods to hash/compare types, which makes it incompatible with STL types. This structes
// helps to resolve needed map type
template <class KeyType, class ValueType>
struct ADTProvider {};

template <class ValueType>
struct ADTProvider<std::string, ValueType> {
    static constexpr char VERIFICATION_METHOD_NAME[] = "full";

    using MapType = std::unordered_map<std::string, ValueType>;
    using SetType = std::unordered_set<std::string>;

    static std::string convertOpToKey(mlir::Operation* op) {
        return stringifyPrimaryLocation(op->getLoc());
    }
};

template <class ValueType>
struct ADTProvider<mlir::Location, ValueType> {
    static constexpr char VERIFICATION_METHOD_NAME[] = "fast";

    using MapType = mlir::DenseMap<mlir::Location, ValueType>;
    using SetType = mlir::DenseSet<mlir::Location>;

    static mlir::Location convertOpToKey(mlir::Operation* op) {
        return op->getLoc();
    }
};

// Utility class to track duplicates for high verbosity messages. Key must be mlir::Location or std::string.
template <class KeyType,
          std::enable_if_t<or_<std::is_same<mlir::Location, KeyType>, std::is_same<std::string, KeyType>>::value,
                           bool> = true>
class HighVerbosityDuplicatesHandler {
    using MapType = typename ADTProvider<KeyType, SmallVector<mlir::Operation*>>::MapType;

public:
    // Update state by insertion of  operation op with key. If handler already has operation with this key, then
    // operation will be marked as duplicate
    void processKey(const KeyType& key, mlir::Operation* op) {
        _key2ops[key].push_back(op);
        if (_key2ops[key].size() == 2) {
            _duplicatedKeys.push_back(key);
        }
    }

    void displayDuplicatesReport() {
        vpux::Logger log = getLogger();
        for (auto&& key : _duplicatedKeys) {
            std::string message;
            if constexpr (PRINT_NAME_DUPLICATES) {
                message = formatv("Found duplicated location '{0}'.", key).str();
            }
            if constexpr (PRINT_OP_DUPLICATES) {
                const auto duplicates = _key2ops[key];
                message += formatv("Duplicated ops of '{0}' are: ", *duplicates.front()).str();
                for (mlir::Operation* op : duplicates) {
                    if (op == duplicates.front()) {
                        continue;
                    }
                    message += formatv("'{0}', ", *op).str();
                }
            }
            if (!message.empty()) {
                log.trace("{0}", message);
            }
        }
    }

private:
    MapType _key2ops;
    SmallVector<KeyType> _duplicatedKeys;
};

// This function verifies uniqueness of all operations within a given operation. Should be called
// after each pass and accept passName to format error message.
template <class KeyType>
mlir::LogicalResult verifyLocationsUniquenessImpl(mlir::Operation* op, StringRef passName) {
    // Resolve container used for verification
    using ADTProvider = ADTProvider<KeyType, std::nullptr_t>;
    constexpr auto verificationMethodName = ADTProvider::VERIFICATION_METHOD_NAME;

    getLogger().trace("Starting {0} verification after {1} pass", verificationMethodName, passName);
    bool isHighVerbosity = isDeveloperBuild();

    typename ADTProvider::SetType uniqLocations;
    size_t totalNamedOps = 0;

    HighVerbosityDuplicatesHandler<KeyType> duplicatesHandler;
    op->walk([&](mlir::Operation* nestedOp) {
        // If we hit operation from late dialect - likely wrong use of setupLocationVerifierPass
        VPUX_THROW_WHEN(isOpFromIgnoredDialect(nestedOp), "Op '{0}' is from ignoring dialect.", nestedOp->getName());

        const auto loc = nestedOp->getLoc();
        // Skip file location from LIT tests
        if (loc.isa<mlir::FileLineColLoc>()) {
            return mlir::WalkResult::advance();
        }

        if (isIgnoredOp(nestedOp)) {
            return mlir::WalkResult::advance();
        }
        VPUX_THROW_UNLESS(loc.isa<mlir::FusedLoc>() || loc.isa<mlir::CallSiteLoc>(),
                          "Location '{0}' isn't FusedLoc after '{1}'", loc, passName);

        ++totalNamedOps;
        const KeyType key = ADTProvider::convertOpToKey(nestedOp);
        uniqLocations.insert(key);
        if (isHighVerbosity) {
            duplicatesHandler.processKey(key, nestedOp);
        }
        return mlir::WalkResult::advance();
    });

    if (totalNamedOps == 0) {
        return mlir::success();
    }

    const size_t numDuplicates = totalNamedOps - uniqLocations.size();
    if (numDuplicates == 0) {
        return mlir::success();
    }
    if (isHighVerbosity) {
        duplicatesHandler.displayDuplicatesReport();
    }

    return errorAt(op, "{0} Pass failed : Found {1} duplicated names after {2} verification", passName, numDuplicates,
                   verificationMethodName);
}

//
// VerifyLocationsInstrumentation
//

class VerifyLocationsInstrumentation final : public mlir::PassInstrumentation {
public:
    void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
        const auto passName = pass->getName();
        if (mlir::failed(verifyLocations(op, passName))) {
            VPUX_THROW("Locations verification failure after '{0}' pass", passName);
        }
    }
};

}  // namespace

LocationsVerificationMode vpux::getLocationsVerificationMode(mlir::ModuleOp moduleOp) {
    if (!moduleOp->hasAttr(LOCATIONS_VERIFICATION_MODE_ATTR_NAME)) {
        return LocationsVerificationMode::OFF;
    }
    auto attr = moduleOp->getAttr(LOCATIONS_VERIFICATION_MODE_ATTR_NAME).dyn_cast<mlir::StringAttr>();
    VPUX_THROW_WHEN(attr == nullptr, "LocationsVerificationModeAttr has wrong type.");
    return symbolizeLocationsVerificationMode(attr.str());
}

LocationsVerificationMode vpux::symbolizeLocationsVerificationMode(StringRef strMode) {
    if (strMode.empty() || strMode == "off") {
        return LocationsVerificationMode::OFF;
    } else if (strMode == "fast") {
        return LocationsVerificationMode::FAST;
    } else if (strMode == "full") {
        return LocationsVerificationMode::FULL;
    } else if (strMode == "thorough") {
        return LocationsVerificationMode::THOROUGH;
    }
    VPUX_THROW("Unknown LocationsVerificationMode '{0}'", strMode);
}

LocationsVerificationMode vpux::getLocationsVerificationMode(
        const mlir::detail::PassOptions::Option<std::string>& locationsVerificationMode) {
    if (!locationsVerificationMode.hasValue()) {
        return LocationsVerificationMode::FAST;
    }
    return vpux::symbolizeLocationsVerificationMode(locationsVerificationMode.getValue());
}

void vpux::setLocationsVerificationMode(mlir::ModuleOp moduleOp, LocationsVerificationMode mode) {
    const auto strMode = stringifyLocationsVerificationMode(mode);
    getLogger().trace("Setting locations verification mode to {0}", strMode);
    if (mode == LocationsVerificationMode::OFF) {
        if (moduleOp->hasAttr(LOCATIONS_VERIFICATION_MODE_ATTR_NAME)) {
            moduleOp->removeAttr(LOCATIONS_VERIFICATION_MODE_ATTR_NAME);
        }
        return;
    }
    auto attr = mlir::StringAttr::get(moduleOp->getContext(), strMode);
    moduleOp->setAttr(LOCATIONS_VERIFICATION_MODE_ATTR_NAME, attr);
}

std::string vpux::stringifyLocationsVerificationMode(LocationsVerificationMode mode) {
    switch (mode) {
    case LocationsVerificationMode::OFF:
        return "off";
    case LocationsVerificationMode::FAST:
        return "fast";
    case LocationsVerificationMode::FULL:
        return "full";
    case LocationsVerificationMode::THOROUGH:
        return "thorough";
    default:
        VPUX_THROW("Unknown LocationsVerificationMode");
    }
}

void vpux::addLocationsVerifier(mlir::PassManager& pm) {
    auto instr = std::make_unique<VerifyLocationsInstrumentation>();
    pm.addInstrumentation(std::move(instr));
}

// Fast verification utilize pointers based comparison, but it may not catch
// some duplicates(different pointers, but same string representation). In the end we want to guarantee uniqueness of
// all names, so must perform full comparison at least in the end of compilation.
mlir::LogicalResult vpux::verifyLocationsUniquenessFull(mlir::Operation* op, StringRef passName) {
    return verifyLocationsUniquenessImpl<std::string>(op, passName);
}

// Compares locations based on their pointers
mlir::LogicalResult vpux::verifyLocationsUniquenessFast(mlir::Operation* op, StringRef passName) {
    return verifyLocationsUniquenessImpl<mlir::Location>(op, passName);
}

mlir::LogicalResult vpux::verifyLocations(mlir::Operation* op, StringRef passName) {
    auto moduleOp = getModuleOp(op);
    const auto mode = vpux::getLocationsVerificationMode(moduleOp);
    if (mode == LocationsVerificationMode::OFF || mode == LocationsVerificationMode::FAST) {
        return mlir::success();
    }

    // We must verify locations within the same operation that the pass was applied to.
    // For exaple:
    // module @TwoFunctions {
    //     func.func @foo1(...) -> ... { ... }
    //     func.func @foo2(...) -> ... { ... }
    //
    //     func.func @main(...) -> ... { ... }
    // }
    // FuncPass is applied to @foo1, @foo2 and @main
    // So when pass is completed for @foo1, we can't check the entire module
    // as the rest of the functions can be changed.
    // This means that FuncPass may lead to duplication within the module,
    // which can only be caught after any next ModulePass.
    if (mode == LocationsVerificationMode::THOROUGH) {
        return vpux::verifyLocationsUniquenessFull(op, passName);
    }
    return vpux::verifyLocationsUniquenessFast(op, passName);
}
