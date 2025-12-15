//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>

#include <iterator>
#include <optional>

namespace vpux {
namespace VPUIP {

constexpr int64_t NPU40XX_SW_KERNEL_ADDRESS_ALIGNMENT = 32;
constexpr size_t MIN_FREE_CYCLES_FOR_PREFETCH_280K = 280000;
constexpr size_t MIN_FREE_CYCLES_FOR_PREFETCH_250K = 250000;

SmallVector<mlir::Attribute> kernelArgsRange(VPUIP::SwKernelOp swKernelOp) {
    SmallVector<mlir::Attribute> attrStorage;

    for (auto&& kernelRun : swKernelOp.getBody().getOps<VPUIP::SwKernelRun>()) {
        if (kernelRun.getAttrs().has_value()) {
            const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
            const auto& attrs = arrayAttrs.getValue();
            for (const auto& attr : attrs) {
                attrStorage.push_back(attr);
            }
        }
    }
    return attrStorage;
}

mlir::ModuleOp getVPUSWModule(mlir::ModuleOp module, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};

    auto innerModule = module.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    // creating VPU.SW module if it is not yet created
    if (!innerModule) {
        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
        innerModule = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx), vpuSwModuleName);
    }
    return innerModule;
}

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, StringRef builtInFunctionName,
                                          const ArrayRef<mlir::Type> inputTypes, StringRef kernelEntryName,
                                          StringRef kernelSourceFileName, StringRef layerName, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);
    auto vpuswModule = getVPUSWModule(module, log);
    // First try is to keep request name, buildin_+VPUOpName. This happens if VPU Op have just one entry_point kernel
    // used in the model. If more VpuOp are present, but with different implementation, then we will create unique name
    // by append to the name the code entry_point string.
    // Move locally to check first original name and after, by appending ".entry_point", that any name already exist and
    // not create new one.
    SmallString builtInFunctionNameFinal(builtInFunctionName);
    // check if this builtInFunction already created - consider names are unique - e.g. no overloads
    if (auto prebuiltFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(builtInFunctionName)) {
        // Check if already created prebuiltFunction refer to the same code entry_point. If this happens, then no new
        // function will be created and will return the old created name.
        const auto prebuiltKernelEntryPoint = prebuiltFunction->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
        if (prebuiltKernelEntryPoint == kernelEntryName) {
            log.trace("Found builtin function: {0}", builtInFunctionName);
            auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionName);
            auto builtInFunction = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {builtInFlatFunction});
            return builtInFunction;
        }
        // In this case builtInFunctionName waa found, but not have same entr_point, so not reffere to same kernel code.
        // It will be check if version with append ".entry_point" already exist.
        builtInFunctionNameFinal.append(".");
        builtInFunctionNameFinal.append(kernelEntryName);
        if (auto prebuiltFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(builtInFunctionNameFinal)) {
            log.trace("Found builtin function: {0}", builtInFunctionNameFinal);
            auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionNameFinal);
            auto builtInFunction = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {builtInFlatFunction});
            return builtInFunction;
        }
    }

    auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionNameFinal);
    auto builtInFunction = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {builtInFlatFunction});

    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, {});

    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto builtInOp = innerModuleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(ctx), builtInFunctionNameFinal,
                                                                   funcType);

    // modifying attributes
    builtInOp.setSymVisibilityAttr(mlir::StringAttr::get(ctx, "private"));

    builtInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(ctx, kernelEntryName));
    builtInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, kernelSourceFileName));
    builtInOp->setAttr("VPU.kernel_name", mlir::StringAttr::get(ctx, layerName));
    builtInOp->setAttr("VPU.task_type",
                       mlir::SymbolRefAttr::get(ctx, VPU::stringifyActShaveTaskType(VPU::ActShaveTaskType::COMPUTE)));

    log.trace("Added new builtin function: {0}", builtInFunctionNameFinal);
    return builtInFunction;
}

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, VPU::LayerOpInterface origOp,
                                          ArrayRef<mlir::Value> operands, ArrayRef<mlir::Value> results,
                                          const VPUIP::KernelInfo& kernelInfo, const Logger& log) {
    OpBuilderLogger builderLog(log);

    SmallString builtInFunctionName{VPUIP::SW_KERNEL_NAME_PREFIX};
    if (auto externalKernelOp = mlir::dyn_cast<VPU::ExternalKernelOp>(origOp.getOperation())) {
        builtInFunctionName.append(externalKernelOp.getUniqueId().str());
    } else {
        auto nonNamespaceOpName = origOp->getName().getStringRef().slice(
                origOp->getName().getDialectNamespace().size() + 1, mlir::StringRef::npos);
        builtInFunctionName.append(nonNamespaceOpName);
    }

    const auto convertToUnrankedTypes = [](mlir::Value operand) -> SmallVector<mlir::Type> {
        SmallVector<mlir::Type> result;
        if (auto type = mlir::dyn_cast_or_null<mlir::MemRefType>(operand.getType())) {
            result.emplace_back(mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace()));
        } else if (auto type = mlir::dyn_cast_or_null<VPUIP::BoundedBufferType>(operand.getType())) {
            const auto dataNDType = mlir::cast<NDTypeInterface>(type.getData());
            result.emplace_back(mlir::UnrankedMemRefType::get(dataNDType.getElementType(), dataNDType.getMemSpace()));
            const auto shapeNDType = mlir::cast<NDTypeInterface>(type.getDynamicShape());
            result.emplace_back(mlir::UnrankedMemRefType::get(shapeNDType.getElementType(), shapeNDType.getMemSpace()));
        } else if (auto type = mlir::dyn_cast_or_null<VPUIP::DistributedBufferType>(operand.getType())) {
            const auto compactType = type.getCompactType();
            result.emplace_back(
                    mlir::UnrankedMemRefType::get(compactType.getElementType(), compactType.getMemorySpace()));
        }
        VPUX_THROW_UNLESS(!result.empty(),
                          "Only MemRef or VPUIP::BoundedBufferType type are supported as createBuiltInFunction "
                          "operands, got: {0}",
                          operand.getType());
        return result;
    };

    auto& args = kernelInfo.args;
    SmallVector<mlir::Type> inputTypes;
    for (const auto& operand : operands) {
        inputTypes.append(convertToUnrankedTypes(operand));
    };
    for (const auto& result : results) {
        inputTypes.append(convertToUnrankedTypes(result));
    };
    std::transform(args.begin(), args.end(), std::back_inserter(inputTypes), [&module](mlir::Attribute arg) {
        const auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(arg);
        return typedAttr != nullptr ? typedAttr.getType() : mlir::NoneType::get(module.getContext());
    });

    return VPUIP::createBuiltInFunction(module, builtInFunctionName, inputTypes, kernelInfo.entryName,
                                        kernelInfo.sourceFileName, kernelInfo.layerName, log);
}

void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log, vpux::config::ArchKind arch) {
    auto vpuswModule = getVPUSWModule(module, log);

    static const SmallString runtimeKernelName{"runtime"};
    static const SmallString runtimeKernelEntryName = static_cast<const SmallString>("nnActEntry");

    // check if runtimeKernel already created
    auto runtimeKernelFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(runtimeKernelName);
    if (runtimeKernelFunction) {
        log.trace("Found builtin function: {0}", runtimeKernelName);
        return;
    }

    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);

    // creating runtime kernel function
    const auto funcType = mlir::FunctionType::get(ctx, {}, {});
    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto runtimeFunctionOp =
            innerModuleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(ctx), runtimeKernelName, funcType);

    // modifying attributes
    runtimeFunctionOp.setSymVisibilityAttr(mlir::StringAttr::get(ctx, "private"));

    runtimeFunctionOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, runtimeKernelEntryName));

    log.trace("Added runtime kernel function: {0}", runtimeKernelEntryName);

    // creating name symbol
    auto runtimeFlatSym = mlir::SymbolRefAttr::get(ctx, runtimeKernelName);
    auto runtimeSym = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {runtimeFlatSym});

    static constexpr int64_t defaultStackSize = 4096;

    // as for now all arches have 2 shaves per tile
    constexpr int nShavePerTile = 2;
    auto tilesUsed = VPUIP::getNumTilesUsed(module);
    auto maxShaves = tilesUsed * nShavePerTile;
    if (arch == vpux::config::ArchKind::NPU40XX) {
        maxShaves = std::min(maxShaves, static_cast<int64_t>(12));
    } else if (arch == vpux::config::ArchKind::NPU50XX) {
        maxShaves = std::min(maxShaves, static_cast<int64_t>(6));
    }
    SmallVector<int64_t> stacksArray(maxShaves, defaultStackSize);

    //  adding runtime kernel configuration - stacks, etc
    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
    moduleBuilder.create<VPURT::SWRunTimeOp>(mlir::UnknownLoc::get(ctx), runtimeSym, getIntArrayAttr(ctx, stacksArray));
}

void initSwKernel(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  ArrayRef<mlir::Attribute> args, const Logger& log, VPUIP::SwKernelRun swKernelRunOp) {
    OpBuilderLogger builderLog(log);
    auto* ctx = swKernelOp.getContext();
    auto& bodyRegion = swKernelOp.getBody();
    auto& swKernelBlock = bodyRegion.emplaceBlock();

    // embedding block args
    auto addBlockArgs = [&swKernelBlock](auto&& cnt) {
        for (auto&& arg : cnt) {
            swKernelBlock.addArgument(arg.getType(), arg.getLoc());
        }
    };

    addBlockArgs(inputs);
    addBlockArgs(outputBuffs);

    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // pack input/outputs and constants into single call to sw_kernel_run
    SmallVector<mlir::Value> operands;
    auto fetchOperands = [&operands](auto&& cnt) {
        for (auto&& arg : cnt) {
            operands.push_back(arg);
        }
    };

    auto blockArgs = swKernelBlock.getArguments();
    fetchOperands(blockArgs);

    auto argsAttr = args.empty() ? nullptr : mlir::ArrayAttr::get(ctx, args);

    if (swKernelRunOp != nullptr) {
        auto numBlockArgs = swKernelBlock.getNumArguments();
        auto numSwKernelRunArgs = swKernelRunOp->getNumOperands();
        VPUX_THROW_UNLESS(numSwKernelRunArgs != 0, "SW Kernel Run has 0 Operands at '{0}'", swKernelOp->getLoc());
        VPUX_THROW_UNLESS(numBlockArgs % numSwKernelRunArgs == 0, "Invalid block arg num at '{0}'",
                          swKernelOp->getLoc());
        auto tileNum = numBlockArgs / numSwKernelRunArgs;

        VPUX_THROW_UNLESS(swKernelOp.getInputs().size() % tileNum == 0 && swKernelOp.getResults().size() % tileNum == 0,
                          "Invalid block arg num at '{0}'", swKernelOp->getLoc());
        auto numSwKernelRunInputs = swKernelOp.getInputs().size() / tileNum;
        auto numSwKernelRunOutputs = swKernelOp.getResults().size() / tileNum;

        if (argsAttr != nullptr) {
            swKernelRunOp.setAttrsAttr(argsAttr);
        }

        for (auto tileIdx : irange(tileNum)) {
            auto newRunOp = swKernelBlockBuilder.clone(*swKernelRunOp.getOperation());
            for (auto argInputIdx : irange(numSwKernelRunInputs)) {
                newRunOp->setOperand(checked_cast<unsigned int>(argInputIdx),
                                     swKernelBlock.getArgument(
                                             checked_cast<unsigned int>(tileIdx * numSwKernelRunInputs + argInputIdx)));
            }

            for (auto argOutputIdx : irange(numSwKernelRunOutputs)) {
                newRunOp->setOperand(
                        checked_cast<unsigned int>(numSwKernelRunInputs + argOutputIdx),
                        swKernelBlock.getArgument(checked_cast<unsigned int>(
                                tileNum * numSwKernelRunInputs + tileIdx * numSwKernelRunOutputs + argOutputIdx)));
            }

            log.trace("create {0}th tile of SwKernelRun {1}", tileIdx, swKernelRunOp);
        }
    } else {
        swKernelBlockBuilder.create<VPUIP::SwKernelRun>(mlir::UnknownLoc::get(ctx), mlir::ValueRange(operands),
                                                        argsAttr);
    }
}

SmallVector<int64_t> reversePermutation(mlir::AffineMap map) {
    const auto origPerm = DimsOrder::fromAffineMap(map).toPermutation();
    SmallVector<int64_t> revPerm(origPerm.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = origPerm.size() - 1 - dstInd;
        revPerm[revSrcInd] = revDstInd;
    }

    return revPerm;
}

// special format of dims/order available only on kernel-FW side
int64_t computeReverseMemDim(mlir::Value tensorArg, int64_t dimIdx) {
    const auto inOrder = DimsOrder::fromValue(tensorArg);
    // Negative value means counting dimension from the end
    if (dimIdx < 0) {
        dimIdx += inOrder.numDims();
    }
    MemDim md = inOrder.toMemDim(Dim(dimIdx));

    const auto shape = getShape(tensorArg);
    auto nDims = checked_cast<uint32_t>(shape.size());
    return nDims - 1 - md.ind();
}

void initSwKernel(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRunOp, const vpux::Logger& log) {
    auto& bodyRegion = swKernelOp.getBody();
    auto& swKernelBlock = bodyRegion.emplaceBlock();

    OpBuilderLogger builderLog(log);
    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // embedding block args
    auto addBlockArgs = [&swKernelBlock](auto&& cnt) {
        for (auto&& arg : cnt) {
            swKernelBlock.addArgument(arg.getType(), arg.getLoc());
        }
    };

    addBlockArgs(swKernelOp.getInputs());
    addBlockArgs(swKernelOp.getOutputBuffs());

    auto numBlockArgs = swKernelBlock.getNumArguments();
    auto numSwKernelRunArgs = swKernelRunOp->getNumOperands();
    VPUX_THROW_UNLESS(numSwKernelRunArgs != 0, "SW Kernel Run has 0 Operands at '{0}'", swKernelOp->getLoc());
    VPUX_THROW_UNLESS(numBlockArgs % numSwKernelRunArgs == 0, "Invalid block arg num at '{0}'", swKernelOp->getLoc());
    auto tileNum = numBlockArgs / numSwKernelRunArgs;

    VPUX_THROW_UNLESS(swKernelOp.getInputs().size() % tileNum == 0 && swKernelOp.getResults().size() % tileNum == 0,
                      "Invalid block arg num at '{0}'", swKernelOp->getLoc());
    auto numSwKernelRunInputs = swKernelOp.getInputs().size() / tileNum;
    auto numSwKernelRunOutputs = swKernelOp.getResults().size() / tileNum;

    // pack input/outputs and constants into several sw_kernel_run calls
    // For example: For Operation that has 2 inputs, 1 output and tile number is 2. After tile it should be like:
    // inputs: [INPUT0_TILE0] as %arg0: First input with 1st tile
    //         [INPUT1_TILE0] as %arg1: Second input with 1st tile
    //         [INPUT0_TILE1] as %arg2: First input with 2nd tile
    //         [INPUT1_TILE1] as %arg3: Second input with 2nd tile
    // outputs:[OUTPUT_TILE0] as %arg4: Output of 1st tile
    //         [OUTPUT_TILE1] as %arg5: Output of 2nd tile
    // Tile 0: VPUIP.SW.Kernel.run {attrs} (%arg0, %arg1, %arg4)
    // Tile 1: VPUIP.SW.Kernel.run {attrs} (%arg2, %arg3, %arg5)
    // For example: For Operation that has 1 input, 2 output and tile number is 2. After tile it should be like:
    // inputs: [INPUT0_TILE0] as %arg0: First input with 1st tile
    //         [INPUT0_TILE1] as %arg1: First input with 2nd tile
    // outputs:[OUTPUT_TILE0] as %arg2: First Output of 1st tile
    //         [OUTPUT_TILE1] as %arg3: Second Output of 1st tile
    //         [OUTPUT_TILE0] as %arg4: First Output of 2nd tile
    //         [OUTPUT_TILE1] as %arg5: Second Output of 2nd tile
    // Tile 0: VPUIP.SW.Kernel.run {attrs} (%arg0, %arg2, %arg3)
    // Tile 1: VPUIP.SW.Kernel.run {attrs} (%arg1, %arg4, %arg5)
    for (auto tileIdx : irange(tileNum)) {
        auto newRunOp = swKernelBlockBuilder.clone(*swKernelRunOp.getOperation());
        for (auto argInputIdx : irange(numSwKernelRunInputs)) {
            newRunOp->setOperand(checked_cast<unsigned int>(argInputIdx),
                                 swKernelBlock.getArgument(
                                         checked_cast<unsigned int>(tileIdx * numSwKernelRunInputs + argInputIdx)));
        }

        for (auto argOutputIdx : irange(numSwKernelRunOutputs)) {
            newRunOp->setOperand(
                    checked_cast<unsigned int>(numSwKernelRunInputs + argOutputIdx),
                    swKernelBlock.getArgument(checked_cast<unsigned int>(
                            tileNum * numSwKernelRunInputs + tileIdx * numSwKernelRunOutputs + argOutputIdx)));
        }

        log.trace("create {0}th tile of SwKernelRun {1}", tileIdx, swKernelRunOp);
    }
}

bool isJitKernelOp(VPUIP::SwKernelOp swKernelOp) {
    auto module = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::FunctionOpInterface>(swKernelOp.getKernelFunctionAttr());
    if (kernelFunc == nullptr) {
        return false;
    }
    return !kernelFunc.isExternal();
}

SmallString getSwKernelEntryName(VPUIP::SwKernelOp swKernelOp) {
    auto module = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::FunctionOpInterface>(swKernelOp.getKernelFunctionAttr());
    VPUX_THROW_WHEN(kernelFunc == nullptr, "Cannot find kernel function symbol at '{0}'", swKernelOp->getLoc());
    if (!kernelFunc.isExternal()) {
        // ShaveCodeGen kernel, just return its name.
        return kernelFunc.getName();
    }
    auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_name");
    // Ensure backward compatibility; kernel_name can be the same as kernel_entry.
    if (kernelEntryPoint == nullptr) {
        kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    }
    VPUX_THROW_WHEN(kernelEntryPoint == nullptr, "Cannot find kernel entry point at '{0}'", swKernelOp->getLoc());
    return kernelEntryPoint.getValue();
}

// Check whether SwKernelOp is activation.
bool isActivationSwKernelOp(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (llvm::find(SW_ACTIVATION_KERNELS, kernelEntryName) != SW_ACTIVATION_KERNELS.end()) {
        return true;
    }
    return false;
}

// Check whether SwKernelOp supports tiling.
bool isSwKernelTilingSupported(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (llvm::find(SW_KERNELS_SUPPORTING_TILING, kernelEntryName) != SW_KERNELS_SUPPORTING_TILING.end()) {
        return true;
    }
    return false;
}

// Check whether SwKernelOp use dpu.
bool isSwKernelUseDpu(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (llvm::find(SW_KERNELS_USE_DPU, kernelEntryName) != SW_KERNELS_USE_DPU.end()) {
        return true;
    }
    return false;
}

bool isDpuShaveKernelType(VPURT::TaskOp taskOp) {
    if (taskOp.getExecutorKind() != VPU::ExecutorKind::SHAVE_ACT) {
        return false;
    }
    auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp());

    if (swKernelOp == nullptr) {
        return false;
    }

    auto module = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelOp.getKernelFunctionAttr());
    if (kernelFunc == nullptr) {
        return false;
    }
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    if (kernelEntryPoint == nullptr) {
        return false;
    }
    if (!isSwKernelUseDpu(swKernelOp)) {
        return false;
    }
    return true;
}

bool isStridedMemPermuteSupported(VPUIP::SwKernelOp swKernelOp) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getOperand(0).getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());
    const auto inOrder = inputType.getDimsOrder();
    const auto outOrder = outputType.getDimsOrder();

    return inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NCHW;
}

// Check whether SwKernelOp support discontinuous input/output.
bool isStridedDataAccessSupported(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    // SubView can be used for Softmax because it is always tilied on the highest dimension.
    if (kernelEntryName == "softmax" ||
        llvm::find(SW_KERNELS_SUPPORTING_STRIDE, kernelEntryName) != SW_KERNELS_SUPPORTING_STRIDE.end() ||
        (kernelEntryName == "reorder" && isStridedMemPermuteSupported(swKernelOp))) {
        return true;
    }
    return false;
}

namespace {

uint64_t getFloatBits(vpux::type::float16 val) {
    return static_cast<uint64_t>(val.to_bits());
}

uint64_t getFloatBits(float val) {
    uint32_t f32Bits = llvm::bit_cast<uint32_t>(val);
    return static_cast<uint64_t>(f32Bits);
}

template <class IT, class OT>
void packAsFpIntoU64(const SmallVector<IT>& values, SmallVector<int64_t>& params) {
    static constexpr uint32_t PACKED_VALUES_COUNT = sizeof(int64_t) / sizeof(OT);
    static constexpr uint64_t bitWidth = sizeof(OT) * CHAR_BIT;
    OT fltValue[PACKED_VALUES_COUNT];
    size_t packIdx = 0;

    auto pack = [](OT fltVals[PACKED_VALUES_COUNT]) -> uint64_t {
        uint64_t ret = 0;
        for (uint32_t i = 0; i < PACKED_VALUES_COUNT; i++) {
            ret |= getFloatBits(fltVals[i]) << (bitWidth * i);
        }
        return ret;
    };

    for (const auto val : values) {
        fltValue[packIdx++] = static_cast<OT>(val);
        if (packIdx == PACKED_VALUES_COUNT) {
            params.push_back(pack(fltValue));
            packIdx = 0;  // reset pack index
        }
    }

    // Store trailing elements
    if (packIdx) {
        // Pad with zeros up to U64 alignment
        while (packIdx < PACKED_VALUES_COUNT) {
            fltValue[packIdx++] = 0;
        }
        params.push_back(pack(fltValue));
    }
}

}  // namespace

void getQuantParamsAttr(mlir::Value qValue, mlir::Type pType, mlir::ArrayAttr& paramsAttr, int64_t tileSize,
                        int64_t tileOffset) {
    SmallVector<double> scales;
    SmallVector<int64_t> zeroes;
    int64_t quantDim = -1;
    const auto qType = mlir::cast<vpux::NDTypeInterface>(qValue.getType()).getElementType();
    mlir::Type storageType;

    if (mlir::isa<mlir::quant::UniformQuantizedType>(qType)) {
        auto quantParams = mlir::cast<mlir::quant::UniformQuantizedType>(qType);
        storageType = quantParams.getStorageType();
        scales = {quantParams.getScale()};
        zeroes = {quantParams.getZeroPoint()};
    } else if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(qType)) {
        auto quantParams = mlir::cast<mlir::quant::UniformQuantizedPerAxisType>(qType);
        storageType = quantParams.getStorageType();
        quantDim = computeReverseMemDim(qValue, quantParams.getQuantizedDimension());
        scales = {quantParams.getScales().begin(), quantParams.getScales().end()};
        zeroes = {quantParams.getZeroPoints().begin(), quantParams.getZeroPoints().end()};
    } else {
        VPUX_THROW("Unsupported quantized type {0}", qType);
    }

    typedef decltype(scales)::value_type TS;
    typedef decltype(zeroes)::value_type TZ;

    // Convert & pack float values into u64 words for serialization

    if (tileSize != 0) {  // Multi-Cluster/Shave tiling context:
        VPUX_THROW_UNLESS(tileOffset + tileSize <= (int64_t)scales.size(), "Slice exceeds full size");
        scales = SmallVector<double>(scales.begin() + tileOffset, scales.begin() + tileOffset + tileSize);
        zeroes = SmallVector<int64_t>(zeroes.begin() + tileOffset, zeroes.begin() + tileOffset + tileSize);
    }

    const auto needsF32Params = storageType.isInteger(16);
    llvm::SmallVector<int64_t> params;
    params.push_back(quantDim);
    params.push_back(scales.size());
    if (pType.isF16() && !needsF32Params) {
        packAsFpIntoU64<TS, vpux::type::float16>(scales, params);
        packAsFpIntoU64<TZ, vpux::type::float16>(zeroes, params);
    } else if (pType.isF32() || needsF32Params) {
        packAsFpIntoU64<TS, float>(scales, params);
        packAsFpIntoU64<TZ, float>(zeroes, params);
    } else {
        pType.dump();
        VPUX_THROW("Supported non-quantized type : f16/f32");
    }
    paramsAttr = getIntArrayAttr(qValue.getContext(), std::move(params));
}

namespace {
// reverse int attribute from the physical order
int64_t reverseMemDim(DimsOrder inOrder, int64_t dimIdx) {
    const auto origPerm = inOrder.toPermutation();
    return origPerm[origPerm.size() - 1 - dimIdx].ind();
}

// reverse int array attribute from the physical order
SmallVector<int64_t> reverseIntArrayAttr(DimsOrder inOrder, mlir::ArrayAttr arrayAttr) {
    const auto origPerm = inOrder.toPermutation();
    const auto origArray = parseIntArrayAttr<int64_t>(arrayAttr);
    SmallVector<int64_t> permArray(arrayAttr.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = dstInd;
        permArray[revDstInd] = origArray[revSrcInd];
    }
    return permArray;
}

// permute int array attribute in the physical order
SmallVector<int64_t> permuteIntArrayAttr(DimsOrder inOrder, ArrayRef<int64_t> origArray) {
    const auto origPerm = inOrder.toPermutation();
    SmallVector<int64_t> permArray(origArray.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = dstInd;
        permArray[revSrcInd] = origArray[revDstInd];
    }
    return permArray;
}

InputTiling backInferInterpolateSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                  Logger& log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();
    const auto inputs = swKernelOp.getInputs();
    auto inOrder = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[0].getType()).getDimsOrder();

    std::optional<SmallVector<int64_t>> coordinatesShape;
    std::optional<SmallVector<int64_t>> lambdasShape;
    if (inputs.size() >= 2) {
        const auto coordinates = inputs[1];
        coordinatesShape = to_small_vector(getShape(coordinates));
    }
    if (inputs.size() >= 3) {
        const auto lambdas = inputs[2];
        lambdasShape = to_small_vector(getShape(lambdas));
    }

    const auto interpolateMode = static_cast<IE::InterpolateMode>(mlir::dyn_cast<mlir::IntegerAttr>(attrs[1]).getInt());
    const auto coordMode = static_cast<IE::InterpolateCoordMode>(mlir::dyn_cast<mlir::IntegerAttr>(attrs[2]).getInt());
    const auto nearestMode =
            static_cast<IE::InterpolateNearestMode>(mlir::dyn_cast<mlir::IntegerAttr>(attrs[3]).getInt());
    const auto initialInputDims = reverseIntArrayAttr(inOrder, mlir::dyn_cast<mlir::ArrayAttr>(attrs[6]));
    const auto initialOutputDims = reverseIntArrayAttr(inOrder, mlir::dyn_cast<mlir::ArrayAttr>(attrs[7]));
    const auto initialInputOffset = reverseIntArrayAttr(inOrder, mlir::dyn_cast<mlir::ArrayAttr>(attrs[10]));
    const auto initialOutputOffset = reverseIntArrayAttr(inOrder, mlir::dyn_cast<mlir::ArrayAttr>(attrs[11]));

    const auto currentInputDims = to_small_vector(getShape(inputs[0]));

    return vpux::backInferInterpolateTile(outputTile, initialInputDims, initialOutputDims, initialInputOffset,
                                          initialOutputOffset, currentInputDims, coordinatesShape, lambdasShape,
                                          interpolateMode, coordMode, nearestMode, log);
}

int64_t convertKernelAxisToOrigAxis(mlir::Value tensorArg, int64_t kernelAxis) {
    const auto shape = getShape(tensorArg);
    // Dims/Order sequence is not same on kernel-FW & compiler side. Convert the axis from kernel to compiler
    // representation.
    auto nDims = checked_cast<uint32_t>(shape.size());

    return nDims - 1 - kernelAxis;
}

InputTiling backInferGatherSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                             Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();
    const auto inputs = swKernelOp.getInputs();

    const auto kernelAxis = mlir::dyn_cast<mlir::IntegerAttr>(attrs[0]).getValue().getSExtValue();
    const auto axisValue = convertKernelAxisToOrigAxis(inputs[0], kernelAxis);
    const auto batchDims = mlir::dyn_cast<mlir::IntegerAttr>(attrs[1]).getValue().getSExtValue();

    const auto origInputShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[0].getType()).getShape();
    const auto origIndicesShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[1].getType()).getShape();

    const auto indicesRank = mlir::dyn_cast<mlir::IntegerAttr>(attrs[2]).getValue().getSExtValue();

    return vpux::backInferGatherTile(outputTile, origInputShape, origIndicesShape, axisValue, batchDims, false,
                                     indicesRank, log);
}

InputTiling backInferGatherNDSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                               Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "GatherND SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(attrs.size() == checked_cast<size_t>(2), "GatherND SwKernelOp should has two attrs '{0}'",
                      swKernelOp);

    const auto inputs = swKernelOp.getInputs();
    const auto origInputShape = getShape(inputs[0]);
    const auto origIndicesShape = getShape(inputs[1]);
    const auto batchDims = mlir::cast<mlir::IntegerAttr>(attrs[0]).getValue().getSExtValue();

    const auto originalShapeAttrVal =
            vpux::extractOriginalShapeAttrFromGatherNDSwOp(mlir::cast<mlir::ArrayAttr>(attrs[1]))
                    .value_or(Shape(origInputShape));

    return vpux::backInferGatherNDTile(outputTile, origInputShape, origIndicesShape, batchDims, originalShapeAttrVal,
                                       log);
}

InputTiling backInferGatherElementsSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                     Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();
    const auto inputs = swKernelOp.getInputs();

    const auto kernelAxis = mlir::dyn_cast<mlir::IntegerAttr>(attrs[0]).getValue().getSExtValue();
    const auto axisValue = convertKernelAxisToOrigAxis(inputs[0], kernelAxis);

    const auto origInputShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[0].getType()).getShape();
    const auto origIndicesShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[1].getType()).getShape();

    return vpux::backInferGatherElementsTile(outputTile, origInputShape, origIndicesShape, axisValue,
                                             origIndicesShape.size(), log);
}

InputTiling backInferGridSampleSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                 Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();

    const auto origInputShape = mlir::cast<vpux::NDTypeInterface>(inputs[0].getType()).getShape();
    const auto origGridShape = mlir::cast<vpux::NDTypeInterface>(inputs[1].getType()).getShape();
    return vpux::backInferGridSampleTile(outputTile, origInputShape, origGridShape, log);
}

InputTiling backInferDeformableConvolutionSwKernelInputTile(VPUIP::SwKernelOp swKernelOp,
                                                            const vpux::TileInfo& outputTile, Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();

    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);

    const auto inputs = swKernelOp.getInputs();
    const auto attrs = swKernelRun.getAttrs().value();

    VPUX_THROW_UNLESS(inputs.size() == 4, "SwKernelOp {0} should have 4 inputs, got '{1}'", swKernelOp, inputs.size());

    VPUX_THROW_UNLESS(attrs.size() == 8, "SwKernelOp {0} should have 8 attributes, got '{1}'", swKernelOp,
                      attrs.size());

    const auto origInputShape = mlir::cast<vpux::NDTypeInterface>(inputs[0].getType()).getShape();
    const auto origOffsetShape = mlir::cast<vpux::NDTypeInterface>(inputs[1].getType()).getShape();
    const auto origKernelShape = mlir::cast<vpux::NDTypeInterface>(inputs[2].getType()).getShape();
    const auto origMaskShape = mlir::cast<vpux::NDTypeInterface>(inputs[3].getType()).getShape();

    auto inOrder = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[0].getType()).getDimsOrder();
    const auto initialOutputOffset = reverseIntArrayAttr(inOrder, mlir::dyn_cast<mlir::ArrayAttr>(attrs[7]));

    return vpux::backInferDeformableConvolutionTile(outputTile, origInputShape, origOffsetShape, origKernelShape,
                                                    origMaskShape, initialOutputOffset, log);
}

InputTiling backInferRMSSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                          Logger /*log*/) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();
    TileInfo gammaTile(getShape(inputs[1]));
    const auto inShape = mlir::cast<vpux::NDTypeInterface>(inputs[0].getType()).getShape();
    auto inTile = outputTile;
    gammaTile.shape[Dim(0)] = inShape[Dim(0)];

    return TilingInfo{{std::move(inTile), std::move(gammaTile)}};
}

InputTiling backInferRoPESwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                           Logger /*log*/) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();
    TileInfo cosTile(getShape(inputs[1]));
    TileInfo sinTile(getShape(inputs[2]));
    auto inTile = outputTile;
    // The Cosine and Sine operations offer flexibility in channel configuration:
    // - Channels: You can choose to match the input's number of channels or set it to 1
    // - Height: Unlike channels, the height for Cosine and Sine operations can differ from the input height
    if (cosTile.shape[Dim(1)] > 1) {
        if (cosTile.shape[Dim(2)] != inTile.shape[Dim(2)]) {
            cosTile.shape[Dim(1)] = inTile.shape[Dim(1)];
            sinTile.shape[Dim(1)] = inTile.shape[Dim(1)];
            sinTile.offsets[Dim(1)] = inTile.offsets[Dim(1)];
            cosTile.offsets[Dim(1)] = inTile.offsets[Dim(1)];
        } else {
            cosTile = inTile;
            sinTile = inTile;
        }
    } else {
        cosTile.shape[Dim(2)] = inTile.shape[Dim(2)];
        sinTile.shape[Dim(2)] = inTile.shape[Dim(2)];
        sinTile.offsets[Dim(2)] = inTile.offsets[Dim(2)];
        cosTile.offsets[Dim(2)] = inTile.offsets[Dim(2)];

        cosTile.shape[Dim(0)] = inTile.shape[Dim(0)];
        sinTile.shape[Dim(0)] = inTile.shape[Dim(0)];
        sinTile.offsets[Dim(0)] = inTile.offsets[Dim(0)];
        cosTile.offsets[Dim(0)] = inTile.offsets[Dim(0)];
    }

    return TilingInfo{{std::move(inTile), std::move(cosTile), std::move(sinTile)}};
}

void adjustMaskOrBiasTile(TileInfo& maskTile, const TileInfo& qTile) {
    bool is2DMask = maskTile.shape[Dims4D::Act::H] != 1;
    if (is2DMask) {
        maskTile.shape[Dims4D::Act::H] = qTile.shape[Dims4D::Act::H];
        maskTile.offsets[Dims4D::Act::H] = qTile.offsets[Dims4D::Act::H];
    }
    bool is3DMask = maskTile.shape[Dims4D::Act::C] != 1;
    if (is3DMask) {
        maskTile.shape[Dims4D::Act::C] = qTile.shape[Dims4D::Act::C];
        maskTile.offsets[Dims4D::Act::C] = qTile.offsets[Dims4D::Act::C];
    }
}

void pushSDPAOptionalInputs(const mlir::OperandRange& inputs, InputTiling& inTiles) {
    int inSize = inputs.size();
    ShapeRef inputVShape = getShape(inputs[2]);
    for (int i = 3; i < inSize - 1; i++) {
        ShapeRef unknownShape = getShape(inputs[i]);
        TileInfo unknownTile(getShape(inputs[i]));
        if (unknownShape[Dims4D::Act::W] == inputVShape[Dims4D::Act::W]) {
            TileInfo inQTile = inTiles.tiles[0];
            adjustMaskOrBiasTile(unknownTile, inQTile);
            inTiles.tiles.push_back(unknownTile);
        } else {
            // Push input Scale
            TileInfo inScaleTile(getShape(inputs[i]));
            inTiles.tiles.push_back(inScaleTile);
        }
    }
}

InputTiling backInferSDPASwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                           Logger /*log*/) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();
    TileInfo inQTile(getShape(inputs[0]));
    TileInfo inKTile(getShape(inputs[1]));
    TileInfo inVTile(getShape(inputs[2]));

    inQTile.shape[Dims4D::Act::H] = outputTile.shape[Dims4D::Act::H];
    inQTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
    inQTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inQTile.offsets[Dims4D::Act::H] = outputTile.offsets[Dims4D::Act::H];
    inQTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
    inQTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];

    inKTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inKTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inKTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
    inKTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];

    inVTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
    inVTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inVTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
    inVTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];

    // InputQ, inputK and InputV are mandatory
    InputTiling inTiles = TilingInfo{{std::move(inQTile), std::move(inKTile), std::move(inVTile)}};
    pushSDPAOptionalInputs(inputs, inTiles);

    // DataStorage is always present because it's generated if absent, at VPU dialect level
    TileInfo dataStorageTile(getShape(inputs[inputs.size() - 1]));
    dataStorageTile.shape[Dims4D::Act::H] = outputTile.shape[Dims4D::Act::H];
    dataStorageTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
    dataStorageTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    dataStorageTile.offsets[Dims4D::Act::H] = outputTile.offsets[Dims4D::Act::H];
    dataStorageTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
    dataStorageTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inTiles.tiles.push_back(dataStorageTile);

    return inTiles;
}

void pushSDPAExtendedOptionalInputs(const mlir::OperandRange& inputs, InputTiling& inTiles, int64_t optionalSizeArea) {
    int inSize = inputs.size();
    ShapeRef inputVShape = getShape(inputs[2]);
    for (int i = 3; i < inSize - optionalSizeArea; i++) {
        ShapeRef unknownShape = getShape(inputs[i]);
        TileInfo unknownTile(getShape(inputs[i]));
        if (unknownShape[Dims4D::Act::W] == inputVShape[Dims4D::Act::W]) {
            TileInfo inQTile = inTiles.tiles[0];
            adjustMaskOrBiasTile(unknownTile, inQTile);
            inTiles.tiles.push_back(unknownTile);
        } else {
            // Push input Scale
            TileInfo inScaleTile(getShape(inputs[i]));
            inTiles.tiles.push_back(inScaleTile);
        }
    }
}

InputTiling backInferSDPAExtendedSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                   Logger /*log*/) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();
    TileInfo inQTile(getShape(inputs[0]));
    TileInfo inKTile(getShape(inputs[1]));
    TileInfo inVTile(getShape(inputs[2]));

    inQTile.shape[Dims4D::Act::H] = outputTile.shape[Dims4D::Act::H];
    inQTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
    inQTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inQTile.offsets[Dims4D::Act::H] = outputTile.offsets[Dims4D::Act::H];
    inQTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
    inQTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];

    inKTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inKTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inKTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
    inKTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];

    inVTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
    inVTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inVTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
    inVTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];

    // InputQ, inputK and InputV are mandatory
    InputTiling inTiles = TilingInfo{{std::move(inQTile), std::move(inKTile), std::move(inVTile)}};

    // Last 2 inputs, DataStorage and DPUStorage are always present because it's generated if absent, in VPU dialect
    pushSDPAExtendedOptionalInputs(inputs, inTiles, 2);

    TileInfo dataStorageTile(getShape(inputs[inputs.size() - 2]));
    dataStorageTile.shape[Dims4D::Act::H] = outputTile.shape[Dims4D::Act::H];
    dataStorageTile.offsets[Dims4D::Act::H] = outputTile.offsets[Dims4D::Act::H];

    inTiles.tiles.push_back(dataStorageTile);

    TileInfo dpuStorageTile(getShape(inputs[inputs.size() - 1]));
    inTiles.tiles.push_back(dpuStorageTile);

    return inTiles;
}

InputTiling backInferDepthToSpaceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                   Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();

    auto inShape = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getInputs()[0].getType()).getShape();
    const auto blockSize = mlir::cast<mlir::IntegerAttr>(attrs[0]).getInt();

    return vpux::backInferDepthToSpaceTile(outputTile, inShape, blockSize, log);
}

InputTiling backInferPadSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();

    const auto origInputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getInputs()[0].getType());
    const auto origInputShape = origInputType.getShape();
    const auto origOutputShape = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getResults()[0].getType()).getShape();
    const auto order = origInputType.getDimsOrder();

    // Padding attr at VPUIP dialect are stored in memory order so convert to default order
    // to be aligned with shape representation
    const auto origPadsBegin = reverseIntArrayAttr(order, mlir::dyn_cast<mlir::ArrayAttr>(attrs[0]));
    const auto origPadsEnd = reverseIntArrayAttr(order, mlir::dyn_cast<mlir::ArrayAttr>(attrs[1]));

    return backInferPadTile(outputTile, origInputShape, origOutputShape, ShapeRef(origPadsBegin), ShapeRef(origPadsEnd),
                            log);
}

InputTiling backInferReduceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                             StringRef kernelEntryName, Logger log) {
    log.trace("Try to back infer input tiling for {0}, output tile: {1}", kernelEntryName, outputTile);

    const auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    const auto numInputs = swKernelOp.getInputs().size();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    VPUX_THROW_UNLESS(numInputs, "SwKernelOp {0} should have 1 input, got '{1}'", swKernelOp, numInputs);

    const auto input = swKernelOp.getOperand(0);
    const auto inputOrder = mlir::cast<vpux::NDTypeInterface>(input.getType()).getDimsOrder();
    const auto inputShape = getShape(input);
    const auto attrs = swKernelRun.getAttrs().value();

    VPUX_THROW_UNLESS(attrs.size() == 3, "SwKernelOp {0} should have 3 attributes, got '{1}'", swKernelOp,
                      attrs.size());
    VPUX_THROW_UNLESS(inputShape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());

    auto inputTile = outputTile;
    const auto reversedAxes = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(attrs[2]));
    for (const auto reversedAxis : reversedAxes) {
        const auto axis = reverseMemDim(inputOrder, reversedAxis);
        const auto d = Dim(axis);
        inputTile.shape[d] = inputShape[d];
    }

    return TilingInfo{std::move(inputTile)};
}

InputTiling backInferMatMulSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                             Logger log) {
    log.trace("Try to back infer input tiling for matmul, output tile: {0}", outputTile);

    const auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    const auto numInputs = swKernelOp.getInputs().size();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    VPUX_THROW_UNLESS(numInputs == 2, "SwKernelOp {0} should have 2 inputs, got '{1}'", swKernelOp, numInputs);

    const auto input1 = swKernelOp.getOperand(0);
    const auto input2 = swKernelOp.getOperand(1);
    const auto input1Shape = getShape(input1);
    const auto input2Shape = getShape(input2);
    const auto attrs = swKernelRun.getAttrs().value();

    VPUX_THROW_UNLESS(attrs.size() == 2, "SwKernelOp {0} should have 2 attributes, got '{1}'", swKernelOp,
                      attrs.size());
    VPUX_THROW_UNLESS(input1Shape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());
    VPUX_THROW_UNLESS(input2Shape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());

    auto input1Tile = outputTile;
    input1Tile.shape[Dim(input1Tile.shape.size() - 2)] = input1Shape[Dim(input1Shape.size() - 2)];
    input1Tile.shape[Dim(input1Tile.shape.size() - 1)] = input1Shape[Dim(input1Shape.size() - 1)];

    auto input2Tile = outputTile;
    input2Tile.shape[Dim(input2Tile.shape.size() - 2)] = input2Shape[Dim(input2Shape.size() - 2)];
    input2Tile.shape[Dim(input2Tile.shape.size() - 1)] = input2Shape[Dim(input2Shape.size() - 1)];

    return InputTiling{{std::move(input1Tile), std::move(input2Tile)}};
}

SmallVector<mlir::Attribute> getDeformableConvolutionSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                                                 ArrayRef<mlir::Attribute> origAttr,
                                                                                 const TileInfo& outTile, Logger log) {
    log.trace("Update attrs for SwKernel Op at '{0}' for out tile {1}", swKernelOp, outTile);

    // Get output tile against the original output

    auto kernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    auto attrs = kernelRun.getAttrs().value();

    VPUX_THROW_UNLESS(origAttr.size() == attrs.size(), "Unmatched attr size found at '{0}'", swKernelOp);

    VPUX_THROW_UNLESS(attrs.size() == 8, "SwKernelOp {0} should have 8 attributes, got '{1}'", swKernelOp,
                      attrs.size());

    SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());

    auto dim = mlir::dyn_cast<vpux::NDTypeInterface>(swKernelOp.getInputs()[0].getType()).getDimsOrder();

    const auto initialOutputOffset = reverseIntArrayAttr(dim, mlir::dyn_cast<mlir::ArrayAttr>(attrs[7]));

    const auto localOutputOffset = to_small_vector(outTile.offsets);

    SmallVector<int64_t> outputTileOffset;

    std::transform(localOutputOffset.begin(), localOutputOffset.end(), initialOutputOffset.begin(),
                   std::back_inserter(outputTileOffset), std::plus<int64_t>());

    auto newOutputTile = outTile;

    newOutputTile.offsets = Shape(outputTileOffset);

    newAttrs[7] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(dim, outputTileOffset));

    return newAttrs;
}

SmallVector<mlir::Attribute> getInterpolateSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                                       ArrayRef<mlir::Attribute> origAttr,
                                                                       const TilingInfo& inputTiling,
                                                                       const TileInfo& outTile, Logger log) {
    log.trace("update attrs for SwKernel Op at '{0}' for out tile {1}", swKernelOp, outTile);
    // Get output tile against the original output
    auto kernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    auto attrs = kernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(origAttr.size() == attrs.size(), "Unmatched attr size found at '{0}'", swKernelOp);

    SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());
    auto dim = mlir::dyn_cast<vpux::NDTypeInterface>(swKernelOp.getInputs()[0].getType()).getDimsOrder();
    TileInfo inputTile = inputTiling.tiles[0];
    const auto initialInputDims = reverseIntArrayAttr(dim, mlir::dyn_cast<mlir::ArrayAttr>(attrs[6]));
    const auto initialOutputDims = reverseIntArrayAttr(dim, mlir::dyn_cast<mlir::ArrayAttr>(attrs[7]));
    const auto initialInputOffset = reverseIntArrayAttr(dim, mlir::dyn_cast<mlir::ArrayAttr>(attrs[10]));
    const auto initialOutputOffset = reverseIntArrayAttr(dim, mlir::dyn_cast<mlir::ArrayAttr>(attrs[11]));
    const auto localInputOffset = to_small_vector(inputTile.offsets);
    const auto localOutputOffset = to_small_vector(outTile.offsets);
    SmallVector<int64_t> inputTileOffset;
    SmallVector<int64_t> outputTileOffset;
    std::transform(localInputOffset.begin(), localInputOffset.end(), initialInputOffset.begin(),
                   std::back_inserter(inputTileOffset), std::plus<int64_t>());
    std::transform(localOutputOffset.begin(), localOutputOffset.end(), initialOutputOffset.begin(),
                   std::back_inserter(outputTileOffset), std::plus<int64_t>());
    auto newInputTiling = inputTiling;
    newInputTiling.tiles[0].offsets = Shape(inputTileOffset);
    auto newOutputTile = outTile;
    newOutputTile.offsets = Shape(outputTileOffset);
    newAttrs[10] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(dim, inputTileOffset));
    newAttrs[11] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(dim, outputTileOffset));
    return newAttrs;
}

SmallVector<mlir::Attribute> getPadSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                               ArrayRef<mlir::Attribute> origAttr,
                                                               const TileInfo& outTile, Logger log) {
    log.trace("update attrs for Pad SwKernel Op at '{0}' for out tile {1}", swKernelOp, outTile);
    auto kernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    auto attrs = kernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(origAttr.size() == attrs.size(), "Unmatched attr size found at '{0}'", swKernelOp);

    SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getResults()[0].getType());
    const auto outShape = outType.getShape();
    auto order = outType.getDimsOrder();

    // Padding attrs at VPUIP dialect are stored in memory-order so convert to default-order
    // to be aligned with shape representation
    auto padsBegin = reverseIntArrayAttr(order, mlir::dyn_cast<mlir::ArrayAttr>(attrs[0]));
    auto padsEnd = reverseIntArrayAttr(order, mlir::dyn_cast<mlir::ArrayAttr>(attrs[1]));

    vpux::updatePadOpAttrsAfterTiling(outShape, outTile, padsBegin, padsEnd);

    // Convert new pads back to memory-order
    newAttrs[0] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(order, padsBegin));
    newAttrs[1] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(order, padsEnd));
    return newAttrs;
}

SmallVector<mlir::Attribute> getDequantizeSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                                      ArrayRef<mlir::Attribute> origAttr,
                                                                      const TileInfo& outTile, Logger log) {
    auto kernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    auto attrs = kernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(origAttr.size() == attrs.size(), "Unmatched attr size found at '{0}'", swKernelOp);

    const auto input = swKernelOp.getInputs()[0];
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto elementType = inType.getElementType();

    Dim quantDim;
    bool attrNeedUpdates = false;
    if (auto quantParams = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
        auto quantAxis = quantParams.getQuantizedDimension();
        quantDim = Dim(quantAxis);
        if (outTile.axis[quantDim] > 1) {
            log.trace("update attrs for Dequantize SwKernel Op at '{0}' for out tile {1}", swKernelOp, outTile);
            attrNeedUpdates = true;
        }
    }

    if (!attrNeedUpdates) {
        return SmallVector<mlir::Attribute>{origAttr};
    }

    const auto oType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getOutputs()[0].getType());
    int64_t sliceSize = outTile.shape[quantDim];
    int64_t sliceOffset = outTile.offsets[quantDim];
    mlir::ArrayAttr paramsAttr;
    getQuantParamsAttr(input, oType.getElementType(), paramsAttr, sliceSize, sliceOffset);

    return SmallVector<mlir::Attribute>{paramsAttr};
}

SmallVector<mlir::Attribute> getLstmSequenceSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                                        ArrayRef<mlir::Attribute> origAttr,
                                                                        const TileInfo& outTile, Logger log) {
    log.trace("Update attrs for LSTMSequence SwKernelOp at '{0}' for out tile {1}", swKernelOp, outTile);

    SmallVector<mlir::Attribute> newAttrs(origAttr.begin(), origAttr.end());
    const auto isTileOverNumDirections = outTile.axis[Dims4D::Act::C] > 1;
    if (!isTileOverNumDirections) {
        return newAttrs;
    }

    // If the operator is tiled along the numDirections dimension, it indicates that it is bidirectional
    // (directionAttr = 2) and is split into one forward (directionAttr = 0) and one reverse (directionAttr = 1)
    // operator. The new attribute value conveniently matches the offset of the numDirections dimension in the output
    // tile.
    const auto numDirectionsDimOffset = outTile.offsets[Dims4D::Act::C];
    const auto newDirectionAttr = getIntAttr(swKernelOp.getContext(), numDirectionsDimOffset);
    newAttrs[0] = newDirectionAttr;

    return newAttrs;
}

SmallVector<mlir::Attribute> getGatherNDSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                                    ArrayRef<mlir::Attribute> origAttr,
                                                                    const TileInfo& outTile, Logger log) {
    log.trace("Update attrs for GatherND SwKernelOp at '{0}' for out tile {1}", swKernelOp, outTile);

    SmallVector<mlir::Attribute> newAttrs(origAttr.begin(), origAttr.end());

    auto originalShape = extractOriginalShapeAttrFromGatherNDSwOp(mlir::dyn_cast<mlir::ArrayAttr>(origAttr[1]));
    if (!originalShape.has_value()) {
        return newAttrs;
    }

    const auto batchDims = mlir::cast<mlir::IntegerAttr>(origAttr[0]).getValue().getSExtValue();
    const auto outTileShape = outTile.shape;

    // Input data with coord part cannot be tiled
    // Only other dimension needs update
    auto newOriginalShape = originalShape.value();
    for (auto idx = 0; idx < batchDims; idx++) {
        newOriginalShape[Dim(idx)] = outTileShape[Dim(idx)];
    }
    newOriginalShape.back() = outTileShape.back();

    auto newOriginalShapeAttr = getIntArrayAttr(swKernelOp.getContext(), newOriginalShape);
    newAttrs[1] = packOriginalShapeAttrForGatherNDSwOp(newOriginalShapeAttr, swKernelOp.getContext());

    return newAttrs;
}

InputTiling backInferTopKSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto inOrder = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getInputs()[0].getType()).getDimsOrder();
    const auto attrs = swKernelRun.getAttrs().value();
    const auto axis = reverseMemDim(inOrder, mlir::cast<mlir::IntegerAttr>(attrs[0]).getInt());

    const auto isLargerThanZero = [](const int64_t dimSize) -> bool {
        return dimSize > 0;
    };

    const auto inShape = getShape(swKernelOp.getInputs()[0]);
    SmallVector<TileInfo> inputTiles;

    VPUX_THROW_UNLESS(inShape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());

    auto curTile = outputTile;
    for (auto ind : irange(inShape.size())) {
        const auto d = Dim(ind);
        if (axis == d.ind()) {
            curTile.shape[d] = inShape[d];
        }
    }
    inputTiles.push_back(curTile);
    if (swKernelOp.getInputs().size() > 1) {
        const auto topKBufferShape = getShape(swKernelOp.getInputs()[1]);
        TileInfo topKBufferTile(topKBufferShape);
        topKBufferTile.shape[Dims4D::Act::W] = topKBufferShape[Dims4D::Act::W] / 2;
        if (llvm::any_of(outputTile.offsets, isLargerThanZero)) {
            topKBufferTile.offsets[Dims4D::Act::W] = topKBufferShape[Dims4D::Act::W] / 2;
        }
        inputTiles.push_back(topKBufferTile);
    }

    return TilingInfo{inputTiles};
}

bool isReduceKernelEntry(StringRef kernelEntryName) {
    static const std::unordered_set<std::string> reduceEntryNames = {
            "reduce_l1",   "reduce_l2",          "reduce_logical_and", "reduce_logical_or", "reduce_max",
            "reduce_mean", "reduce_mean_square", "reduce_min",         "reduce_prod",       "reduce_sum"};

    return reduceEntryNames.find(kernelEntryName.str()) != reduceEntryNames.end();
}

InputTiling backInferGRUSequenceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTileY,
                                                  Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();

    const auto origInputShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[0].getType()).getShape();
    const auto origInitialHiddenStateShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[1].getType()).getShape();
    const auto origWShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[2].getType()).getShape();
    const auto origRShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[3].getType()).getShape();
    const auto origBShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[4].getType()).getShape();

    TileInfo inputTile(origInputShape);
    TileInfo initialHiddenStateTile(origInitialHiddenStateShape);
    TileInfo wTile(origWShape);
    TileInfo rTile(origRShape);
    TileInfo bTile(origBShape);

    inputTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    inputTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    initialHiddenStateTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    initialHiddenStateTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    return InputTiling{{std::move(inputTile), std::move(initialHiddenStateTile), std::move(wTile), std::move(rTile),
                        std::move(bTile)}};
}

InputTiling backInferGRUSequenceLastPartSwKernelInputTile(VPUIP::SwKernelOp swKernelOp,
                                                          const vpux::TileInfo& outputTileY, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();

    const auto origInputShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[0].getType()).getShape();
    const auto origInitialHiddenStateShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[1].getType()).getShape();
    const auto origRShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[2].getType()).getShape();
    const auto origBShape = mlir::dyn_cast<vpux::NDTypeInterface>(inputs[3].getType()).getShape();

    TileInfo inputTile(origInputShape);
    TileInfo initialHiddenStateTile(origInitialHiddenStateShape);
    TileInfo rTile(origRShape);
    TileInfo bTile(origBShape);

    inputTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    inputTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    initialHiddenStateTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    initialHiddenStateTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    return InputTiling{{std::move(inputTile), std::move(initialHiddenStateTile), std::move(rTile), std::move(bTile)}};
}

InputTiling backInferLSTMGatesSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    SmallVector<TileInfo> inputTiles;
    for (const auto& origInput : swKernelOp.getInputs()) {
        const auto curShape = getShape(origInput);
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                          swKernelOp->getName(), swKernelOp->getLoc());
        auto curTile = outputTile;
        curTile.shape[Dim(curShape.size() - 1)] = curShape[Dim(curShape.size() - 1)];
        inputTiles.push_back(curTile);
    }
    return TilingInfo{inputTiles};
}

InputTiling backInferGRUGatesSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    SmallVector<TileInfo> inputTiles;

    auto curTile = outputTile;
    auto curShape = getShape(swKernelOp.getInputs()[0]);
    curTile.shape[Dim(curShape.size() - 1)] = curShape[Dim(curShape.size() - 1)];
    inputTiles.push_back(curTile);

    curTile = outputTile;
    curShape = getShape(swKernelOp.getInputs()[1]);
    curTile.shape[Dim(curShape.size() - 1)] = curShape[Dim(curShape.size() - 1)];
    inputTiles.push_back(curTile);

    curTile = outputTile;
    curShape = getShape(swKernelOp.getInputs()[2]);
    curTile.shape[Dim(curShape.size() - 1)] = curShape[Dim(curShape.size() - 1)];
    inputTiles.push_back(curTile);

    curShape = getShape(swKernelOp.getInputs()[3]);
    TileInfo bTile(curShape);
    inputTiles.push_back(bTile);

    return TilingInfo{inputTiles};
}

InputTiling backInferLSTMCellSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                               Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    SmallVector<TileInfo> inputTiles;

    // inputs
    const auto inputData = swKernelOp.getInputs()[0];
    const auto initialHiddenState = swKernelOp.getInputs()[1];
    //  weight
    const auto weights = swKernelOp.getInputs()[3];
    const auto weightsHidden = swKernelOp.getInputs()[4];
    const auto biases = swKernelOp.getInputs()[5];

    const auto inputDataShape = getShape(inputData);
    const auto initialHiddenStateShape = getShape(initialHiddenState);

    const auto weightsShape = getShape(weights);
    const auto weightsHiddenShape = getShape(weightsHidden);
    const auto biasesShape = getShape(biases);

    TileInfo inputDataTile(inputDataShape);
    TileInfo initialHiddenStateTile(initialHiddenStateShape);
    TileInfo initialCellStateTile = outputTile;

    TileInfo weightsTile(weightsShape);
    weightsTile.shape[Dim(weightsShape.size() - 2)] = outputTile.shape.back();
    weightsTile.offsets[Dim(weightsShape.size() - 2)] = outputTile.offsets.back();
    weightsTile.axis[Dim(weightsShape.size() - 2)] = outputTile.axis.back();

    TileInfo weightsHiddenTile(weightsHiddenShape);
    weightsHiddenTile.shape[Dim(weightsHiddenShape.size() - 2)] = outputTile.shape.back();
    weightsHiddenTile.offsets[Dim(weightsHiddenShape.size() - 2)] = outputTile.offsets.back();
    weightsHiddenTile.axis[Dim(weightsHiddenShape.size() - 2)] = outputTile.axis.back();

    TileInfo biasesTile(biasesShape);
    biasesTile.shape[Dim(biasesShape.size() - 1)] = outputTile.shape.back();
    biasesTile.offsets[Dim(biasesShape.size() - 1)] = outputTile.offsets.back();
    biasesTile.axis[Dim(biasesShape.size() - 1)] = outputTile.axis.back();

    inputTiles.push_back(inputDataTile);
    inputTiles.push_back(initialHiddenStateTile);
    inputTiles.push_back(initialCellStateTile);
    inputTiles.push_back(weightsTile);
    inputTiles.push_back(weightsHiddenTile);
    inputTiles.push_back(biasesTile);

    log.trace("backInferLSTMCellSwKernelInputTile  outputTile '{0}'", outputTile);
    log.trace("backInferLSTMCellSwKernelInputTile  inputTiles '{0}'", inputTiles);

    return TilingInfo{inputTiles};
}

InputTiling backInferLSTMSequenceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                   Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    // inputs
    const auto inputData = swKernelOp.getInputs()[0];
    const auto initialHiddenState = swKernelOp.getInputs()[1];
    const auto initialCellState = swKernelOp.getInputs()[2];
    const auto weightsHidden = swKernelOp.getInputs()[3];
    const auto biases = swKernelOp.getInputs()[4];
    const auto syncBuffer = swKernelOp.getInputs()[5];

    const auto inputDataShape = getShape(inputData);
    const auto initialHiddenStateShape = getShape(initialHiddenState);
    const auto initialCellStateShape = getShape(initialCellState);
    const auto weightsHiddenShape = getShape(weightsHidden);
    const auto biasesShape = getShape(biases);
    const auto syncBufferShape = getShape(syncBuffer);

    TileInfo inputDataTile(inputDataShape);
    TileInfo initialHiddenStateTile(initialHiddenStateShape);
    TileInfo initialCellStateTile(initialCellStateShape);
    TileInfo weightsHiddenTile(weightsHiddenShape);
    TileInfo biasesTile(biasesShape);
    TileInfo syncBufferTile(syncBufferShape);

    const auto batchSize = outputTile.shape[Dims4D::Act::N];
    const auto batchOffset = outputTile.offsets[Dims4D::Act::N];
    const auto batchAxis = outputTile.axis[Dims4D::Act::N];
    const auto numDirections = outputTile.shape[Dims4D::Act::C];
    const auto numDirectionsOffset = outputTile.offsets[Dims4D::Act::C];
    const auto numDirectionsAxis = outputTile.axis[Dims4D::Act::C];

    inputDataTile.shape[Dims4D::Act::N] = batchSize;
    inputDataTile.shape[Dims4D::Act::C] = numDirections;
    inputDataTile.offsets[Dims4D::Act::N] = batchOffset;
    inputDataTile.offsets[Dims4D::Act::C] = numDirectionsOffset;
    inputDataTile.axis[Dims4D::Act::N] = batchAxis;
    inputDataTile.axis[Dims4D::Act::C] = numDirectionsAxis;

    initialHiddenStateTile.shape[Dims4D::Act::N] = batchSize;
    initialHiddenStateTile.shape[Dims4D::Act::C] = numDirections;
    initialHiddenStateTile.offsets[Dims4D::Act::N] = batchOffset;
    initialHiddenStateTile.offsets[Dims4D::Act::C] = numDirectionsOffset;
    initialHiddenStateTile.axis[Dims4D::Act::N] = batchAxis;
    initialHiddenStateTile.axis[Dims4D::Act::C] = numDirectionsAxis;

    initialCellStateTile.shape[Dims4D::Act::N] = batchSize;
    initialCellStateTile.shape[Dims4D::Act::C] = numDirections;
    initialCellStateTile.offsets[Dims4D::Act::N] = batchOffset;
    initialCellStateTile.offsets[Dims4D::Act::C] = numDirectionsOffset;
    initialCellStateTile.axis[Dims4D::Act::N] = batchAxis;
    initialCellStateTile.axis[Dims4D::Act::C] = numDirectionsAxis;

    weightsHiddenTile.shape[Dims4D::Act::N] = numDirections;
    weightsHiddenTile.offsets[Dims4D::Act::N] = numDirectionsOffset;
    weightsHiddenTile.axis[Dims4D::Act::N] = numDirectionsAxis;

    biasesTile.shape[Dims4D::Act::C] = numDirections;
    biasesTile.offsets[Dims4D::Act::C] = numDirectionsOffset;
    biasesTile.axis[Dims4D::Act::C] = numDirectionsAxis;

    const SmallVector<TileInfo> inputTiles = {std::move(inputDataTile),        std::move(initialHiddenStateTile),
                                              std::move(initialCellStateTile), std::move(weightsHiddenTile),
                                              std::move(biasesTile),           std::move(syncBufferTile)};

    log.trace("backInferLSTMSequenceSwKernelInputTile  outputTile '{0}'", outputTile);
    log.trace("backInferLSTMSequenceSwKernelInputTile  inputTiles '{0}'", inputTiles);

    return TilingInfo{inputTiles};
}

InputTiling backInferRandomUniformSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                    Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    SmallVector<TileInfo> inputTiles;
    for (const auto& origInput : swKernelOp.getInputs()) {
        const auto curShape = getShape(origInput);
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                          swKernelOp->getName(), swKernelOp->getLoc());

        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            curTile.shape[d] = 1;
            curTile.offsets[d] = 0;
            curTile.axis[d] = 1;
        }

        inputTiles.push_back(curTile);
    }

    return TilingInfo{inputTiles};
}

InputTiling backInferRollSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    const auto shiftShape = getShape(swKernelOp.getInputs()[1]);
    const auto axesShape = getShape(swKernelOp.getInputs()[2]);
    TileInfo shiftTile(shiftShape);
    TileInfo axesTile(axesShape);

    return InputTiling{{outputTile, std::move(shiftTile), std::move(axesTile)}};
}

InputTiling backInferMemPermuteSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                 Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    const auto reversedMemPermAttr = swKernelRun.getAttrs().value()[0];
    // sw kernel uses reversed MemPerm reversing formula : newDim = maxDim - currentDim -1
    // IF we want to get original dim : currentDim = maxDim -newDim -1 ( so same formula, we can reverse the
    // reversedMemPerm)
    const auto reversedMemPerm = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(reversedMemPermAttr));
    auto ctx = swKernelOp->getContext();
    const auto reversedAffineMapMemPerm = mlir::AffineMap::getPermutationMap(reversedMemPerm, ctx);
    const auto originalMemPerm = reversePermutation(reversedAffineMapMemPerm);
    // Logic below is same with transformations.cpp::prepareMemPermuteSwap, explained in more detail there.
    const auto originalMemPermMap = mlir::AffineMap::getPermutationMap(originalMemPerm, ctx);

    const auto inputLayoutMap =
            mlir::cast<NDTypeInterface>(swKernelOp.getInputs()[0].getType()).getDimsOrder().toAffineMap(ctx);
    const auto outputLayoutMap =
            mlir::cast<NDTypeInterface>(swKernelOp.getOutputs()[0].getType()).getDimsOrder().toAffineMap(ctx);

    auto reverseInputLayoutMap = mlir::inversePermutation(inputLayoutMap);

    const auto mappingFromOutputToInputMemoryLayout = mlir::inversePermutation(originalMemPermMap);
    const auto mappingOutputToInputLogicalLayout =
            reverseInputLayoutMap.compose(mappingFromOutputToInputMemoryLayout).compose(outputLayoutMap);
    // Inverse the permutation to get the order of dimensions in the input tile.
    const auto permutationOrder = DimsOrder::fromAffineMap(mlir::inversePermutation(mappingOutputToInputLogicalLayout));

    auto curTile = outputTile;

    for (auto ind : irange(outputTile.shape.size())) {
        const auto d = Dim(ind);
        curTile.shape[permutationOrder.dimAt(d.ind())] = outputTile.shape[d];
        curTile.axis[permutationOrder.dimAt(d.ind())] = outputTile.axis[d];
        curTile.offsets[permutationOrder.dimAt(d.ind())] = outputTile.offsets[d];
    }

    return InputTiling{curTile};
}

InputTiling backInferMvn1SumSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto tileAxis = outputTile.axis;
    auto tilingDims = getNonOneDim(tileAxis);
    VPUX_THROW_UNLESS(tilingDims.size() == 1 && tileAxis.size() == 4,
                      "Only support 4D tensor shape with one dim tiling");
    auto tilingDim = tilingDims.front();

    const auto inShape = getShape(swKernelOp.getInputs()[0]);
    const auto outShape = getShape(swKernelOp.getResult(0));

    TileInfo inputTile(inShape);
    if (tilingDim == Dims4D::Act::N || tilingDim == Dims4D::Act::C) {
        inputTile.shape[tilingDim] = outputTile.shape[tilingDim];
        inputTile.offsets[tilingDim] = outputTile.offsets[tilingDim];
        return TilingInfo{std::move(inputTile)};
    }

    // When tiling at the height dimension, it is a very specific operation
    // where any input H size can only produce one line of output H at each shave excluster
    // The important thing is to establish a rule that can infer the input shape from the output tile shape (1)
    // Here, the same rule as with TileActShavePass and UnrollDistributedOpsPass is maintained
    //
    // For example, using 3 clusters and 6 shaves, with an output height of 6 and an input height of 76:
    // If subview is used or MC & MS are tiling on the same dimension,
    // The splitInShapeH list would be:
    //   [Tile0(13), Tile1(13), Tile2(13), Tile3(12), Tile4(13), Tile5(12)]
    // The inTileOffsetH list would be:
    //   [Tile0(0), Tile1(13), Tile2(26), Tile3(39), Tile4(51), Tile5(64)]
    // The index distribution looks like:
    //         SHV0        SHV1
    //   CL0  [Tile0(13)   Tile1(13)]
    //   CL1  [Tile2(13)   Tile3(12)]
    //   CL2  [Tile4(13)   Tile5(12)]
    // At TileActShavePass (outTileShapeH > 1), the Input Tile Info for all clusters at each shave is:
    //   SHV0: data_size: Tile0 + Tile2 + Tile4; data_offset: 0
    //   SHV1: data_size: Tile1 + Tile3 + Tile5; data_offset: Tile0 + Tile2 + Tile4
    // At UnrollDistributedOpsPass (outTileShapeH == 1), the Input Tile Info for each shave and each cluster is:
    //   Directly get values from splitInShape and inTileOffsetH through the index.

    const auto inH = inShape[Dims4D::Act::H];
    const auto outH = outShape[Dims4D::Act::H];
    auto largeNumb = static_cast<int64_t>(inH % outH);
    auto baseSize = static_cast<int64_t>(inH / outH);

    SmallVector<int64_t> splitInShape(outH, baseSize);

    // keep first shave indices equal for correct unroll cmx input buffer offsets
    // Distribute on first shave if largeNumb is greater than outH / 2
    int64_t startIdx = (largeNumb > outH / 2) ? 0 : 1;
    int64_t idx = startIdx;
    for (int64_t replaced = 0; replaced < largeNumb; replaced++) {
        splitInShape[idx] += 1;
        idx += 2;

        // Distribute remaining values to the second shave
        if (idx >= outH) {
            idx = 1;
        }
    }

    SmallVector<int64_t> splitInOffset(outH);
    splitInOffset[0] = 0;
    std::partial_sum(splitInShape.begin(), splitInShape.end() - 1, splitInOffset.begin() + 1);

    auto outTileShapeH = outputTile.shape[Dims4D::Act::H];
    auto outTileOffsetH = outputTile.offsets[Dims4D::Act::H];

    auto inTileShapeH = 0;
    auto inTileOffsetH = 0;
    if (outTileShapeH == 1) {
        inTileShapeH = splitInShape[outTileOffsetH];
        inTileOffsetH = splitInOffset[outTileOffsetH];
    } else {
        const auto isLastShaveData = (outTileOffsetH + outTileShapeH == outH);
        int64_t firstShaveSize = std::accumulate(splitInShape.begin(), splitInShape.end(), 0,
                                                 [index = 0](int64_t acc, int64_t val) mutable {
                                                     return (index++ % 2 == 0) ? acc + val : acc;
                                                 });
        inTileShapeH = isLastShaveData ? inH - firstShaveSize : firstShaveSize;
        inTileOffsetH = isLastShaveData ? firstShaveSize : 0;
    }

    inputTile.shape[Dims4D::Act::H] = inTileShapeH;
    inputTile.offsets[Dims4D::Act::H] = inTileOffsetH;

    return TilingInfo{std::move(inputTile)};
}

InputTiling backInferMvn1NormSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();

    const auto origMeanVarType = mlir::cast<vpux::NDTypeInterface>(swKernelOp.getInputs()[1].getType());
    const auto origMeanVarShape = origMeanVarType.getShape();

    TileInfo inDataTile(outputTile);
    TileInfo inMeanVarTile(origMeanVarShape);

    const auto acrossChannels = mlir::cast<mlir::BoolAttr>(attrs[0]).getValue();
    if (!acrossChannels) {
        inMeanVarTile.shape[Dims4D::Act::C] = inDataTile.shape[Dims4D::Act::C];
        inMeanVarTile.offsets[Dims4D::Act::C] = inDataTile.offsets[Dims4D::Act::C];
    }
    inMeanVarTile.shape[Dims4D::Act::N] = inDataTile.shape[Dims4D::Act::N];
    inMeanVarTile.offsets[Dims4D::Act::N] = inDataTile.offsets[Dims4D::Act::N];

    return TilingInfo{{std::move(inDataTile), std::move(inMeanVarTile)}};
}

InputTiling backInferFlashSDPASwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                Logger) {
    const auto keyShape = getShape(swKernelOp->getOperand(1));
    const auto loc = swKernelOp->getLoc();

    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "Got multiple SwKernel.run operation before multi-shave tiling at {0}", loc);

    auto swKernelRun = *swKernelRuns.begin();

    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);

    // for some reason attributes are stored as [[INT64_MAX, inputMask]] <- array of arrays
    // find out if attenion mask and/or scale tensors are present and get their shape
    const auto attrs = parseIntArrayOfArrayAttr<int64_t>(swKernelRun.getAttrs().value());
    const auto inputsMask = attrs[0][1];

    const auto hasAttentionMask = static_cast<bool>(inputsMask & (1 << 2));
    const auto hasScale = static_cast<bool>(inputsMask & (1 << 3));

    auto attentionMaskShape = std::optional<ShapeRef>{};
    auto scaleShape = std::optional<ShapeRef>{};
    const auto attentionMaskIndex = 10;

    if (hasAttentionMask) {
        attentionMaskShape = getShape(swKernelOp->getOperand(attentionMaskIndex));
    }

    if (hasScale) {
        auto scaleIndex = attentionMaskIndex + hasAttentionMask;
        scaleShape = getShape(swKernelOp->getOperand(scaleIndex));
    }

    auto dpuDescriptorBufferShape = getShape(swKernelOp.getOperand(4));
    auto weightsTable0Shape = getShape(swKernelOp.getOperand(5));
    auto weightsTable1Shape = getShape(swKernelOp.getOperand(6));

    auto inputTiling =
            vpux::VPU::FlashSDPAOpInputTiling(outputTile, keyShape, attentionMaskShape, scaleShape,
                                              dpuDescriptorBufferShape, weightsTable0Shape, weightsTable1Shape);

    auto& dpuDescriptorsBufferTile = inputTiling.tiles[4];
    VPUX_THROW_UNLESS(
            dpuDescriptorsBufferTile.shape[Dims4D::Act::H] % 2 == 0,
            "Can't tile DpuDescriptorsBuffer on SHAVEs. Shape '{0}' is not divisible on 2 on Height dimension at {1}",
            dpuDescriptorsBufferTile.shape, swKernelOp->getLoc());

    dpuDescriptorsBufferTile.shape[Dims4D::Act::H] /= 2;

    if (outputTile.offsets != Shape{0, 0, 0, 0}) {
        dpuDescriptorsBufferTile.offsets[Dims4D::Act::H] = dpuDescriptorsBufferTile.shape[Dims4D::Act::H];
    }

    return inputTiling;
}

InputTiling backInferYUVToRGBSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto H = Dim(1), W = Dim(2), C = Dim(3);  // N = Dim(0)

    VPUX_THROW_UNLESS(outputTile.shape[H] % 2 == 0 && outputTile.shape[W] % 2 == 0,
                      "Invalid YuvToRgbOp outputTile, output C,H channels are not even");

    const auto inputs = swKernelOp.getInputs();
    const auto singlePlane = (inputs.size() == 1);

    if (!singlePlane) {
        if (inputs.size() == 2) {
            // NV12 format: Y plane + UV plane
            TileInfo input1Tile = outputTile;  // Y plane
            TileInfo input2Tile = outputTile;  // UV plane

            input1Tile.shape[C] = 1;                            // Y plane has 1 channel
            input2Tile.shape[C] = 2;                            // UV plane has 2 channels (interleaved U,V)
            input2Tile.shape[H] = outputTile.shape[H] / 2;      // UV plane height is half
            input2Tile.shape[W] = outputTile.shape[W] / 2;      // UV plane width is half
            input2Tile.offsets[H] = outputTile.offsets[H] / 2;  // UV plane offset H is half
            input2Tile.offsets[W] = outputTile.offsets[W] / 2;  // UV plane offset W is half

            return TilingInfo{{std::move(input1Tile), std::move(input2Tile)}};
        } else if (inputs.size() == 3) {
            // I420 format: Y plane + U plane + V plane
            TileInfo input1Tile = outputTile;  // Y plane
            TileInfo input2Tile = outputTile;  // U plane
            TileInfo input3Tile = outputTile;  // V plane

            input1Tile.shape[C] = 1;                            // Y plane has 1 channel
            input2Tile.shape[C] = 1;                            // U plane has 1 channel
            input2Tile.shape[H] = outputTile.shape[H] / 2;      // U plane height is half
            input2Tile.shape[W] = outputTile.shape[W] / 2;      // U plane width is half
            input2Tile.offsets[H] = outputTile.offsets[H] / 2;  // U plane offset H is half
            input2Tile.offsets[W] = outputTile.offsets[W] / 2;  // U plane offset W is half

            input3Tile.shape[C] = 1;                            // V plane has 1 channel
            input3Tile.shape[H] = outputTile.shape[H] / 2;      // V plane height is half
            input3Tile.shape[W] = outputTile.shape[W] / 2;      // V plane width is half
            input3Tile.offsets[H] = outputTile.offsets[H] / 2;  // V plane offset H is half
            input3Tile.offsets[W] = outputTile.offsets[W] / 2;  // V plane offset W is half

            return TilingInfo{{std::move(input1Tile), std::move(input2Tile), std::move(input3Tile)}};
        } else {
            VPUX_THROW("YuvToRGB expects 2 inputs (NV12) or 3 inputs (I420) for multi-plane, got {0}", inputs.size());
        }
    } else {
        VPUX_THROW("YuvToRGB single plane MC/MS is not yet supported");
    }
}

}  // namespace

InputTiling backInferSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const SmallVector<vpux::TileInfo>& outputTiles,
                                       int tileId, Logger log) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    const auto arch = config::getArch(swKernelOp);
    const auto& outputTile = outputTiles[tileId];
    if (kernelEntryName == "interpolate") {
        return backInferInterpolateSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "topk") {
        return backInferTopKSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gather") {
        return backInferGatherSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gatherND") {
        return backInferGatherNDSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "grid_sample") {
        return backInferGridSampleSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "deformable_convolution") {
        return backInferDeformableConvolutionSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gather_elements") {
        return backInferGatherElementsSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "rms_norm") {
        return backInferRMSSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "rope") {
        return backInferRoPESwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "sdpa") {
        return backInferSDPASwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "sdpa_extended") {
        return backInferSDPAExtendedSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "pad") {
        return backInferPadSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "mvn1_sum") {
        return backInferMvn1SumSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "mvn1_norm") {
        return backInferMvn1NormSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "depth_to_space") {
        return backInferDepthToSpaceSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gru_sequence") {
        return backInferGRUSequenceSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gru_sequence_last_part") {
        return backInferGRUSequenceLastPartSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (isReduceKernelEntry(kernelEntryName)) {
        return backInferReduceSwKernelInputTile(swKernelOp, outputTile, kernelEntryName, log);
    } else if (kernelEntryName == "matmul") {
        return backInferMatMulSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "lstm_gates") {
        return backInferLSTMGatesSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gru_gates") {
        return backInferGRUGatesSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "lstm_cell") {
        return backInferLSTMCellSwKernelInputTile(swKernelOp, outputTile, log);
    } else if ((kernelEntryName == "lstm_sequence") || (kernelEntryName == "lstm_dpu")) {
        return backInferLSTMSequenceSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "random_uniform") {
        return backInferRandomUniformSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "roll") {
        return backInferRollSwKernelInputTile(swKernelOp, outputTile, log);
    } else if ((kernelEntryName == "detection_output_sort") && (arch == config::ArchKind::NPU37XX)) {
        return vpux::VPU::DetectionOutputSortOpInputTilingOnShave(swKernelOp, outputTile, tileId, outputTiles.size(),
                                                                  log);
    } else if (kernelEntryName == "reorder") {
        return backInferMemPermuteSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "flash_sdpa") {
        return backInferFlashSDPASwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "nv12_to_rgb" || kernelEntryName == "i420_to_rgb") {
        return backInferYUVToRGBSwKernelInputTile(swKernelOp, outputTile, log);
    }

    SmallVector<TileInfo> inputTiles;
    for (const auto& origInput : swKernelOp.getInputs()) {
        const auto curShape = getShape(origInput);
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                          swKernelOp->getName(), swKernelOp->getLoc());

        // Handle broadcasted inputs
        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            if (curShape[d] == 1) {
                curTile.shape[d] = 1;
                curTile.offsets[d] = 0;
            }
        }

        inputTiles.push_back(curTile);
    }
    return TilingInfo{inputTiles};
}

SmallVector<mlir::Attribute> getSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                            ArrayRef<mlir::Attribute> origAttr,
                                                            const TilingInfo& inputTiling, const TileInfo& outTile,
                                                            Logger log) {
    log.trace("Update SwKernel attrs after tiling at '{0}'", swKernelOp->getLoc());
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "interpolate") {
        return getInterpolateSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, inputTiling, outTile, log);
    } else if (kernelEntryName == "pad") {
        return getPadSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, outTile, log);
    } else if ((kernelEntryName == "lstm_sequence") || (kernelEntryName == "lstm_dpu")) {
        return getLstmSequenceSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, outTile, log);
    } else if (kernelEntryName == "gatherND") {
        return getGatherNDSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, outTile, log);
    } else if (kernelEntryName == "deformable_convolution") {
        return getDeformableConvolutionSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, outTile, log);
    } else if (kernelEntryName == "dequantize") {
        return getDequantizeSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, outTile, log);
    } else {
        return SmallVector<mlir::Attribute>(origAttr.begin(), origAttr.end());
    }
}

SmallVector<int64_t> getPopulateWeightTableSwKernelEntries(VPUIP::SwKernelOp swKernelOp) {
    SmallVector<int64_t> weightsPerClusterPtrs;
    if (!swKernelOp->hasAttr(vpux::VPUIP::weightsPtrsPerClusterAttr)) {
        return weightsPerClusterPtrs;
    }

    return parseIntArrayAttr<int64_t>(
            swKernelOp->getAttrOfType<mlir::ArrayAttr>(vpux::VPUIP::weightsPtrsPerClusterAttr));
}

void updatePopulateWeightTableSwKernel(VPUIP::SwKernelOp swKernelOp, int64_t currOffset, Logger log) {
    log.trace("Update offsets for SwKernel Op at '{0}'", swKernelOp);

    auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    for (auto entry : swKernelRun | indexed) {
        auto attrs = entry.value().getAttrs().value();
        SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());
        const auto newOffset = mlir::cast<mlir::IntegerAttr>(newAttrs[0]).getInt() + currOffset;
        newAttrs[0] = getIntAttr(swKernelOp->getContext(), newOffset);
        entry.value().setAttrsAttr(mlir::ArrayAttr::get(swKernelOp->getContext(), newAttrs));
        log.trace("Updated base offset to {0}", newOffset);
    }
    log.trace("update offsets for SwKernel Op at '{0}' {1}", swKernelOp, currOffset);
}

// Return all tensor types of SwKernelOp that will be tiled
SmallVector<vpux::NDTypeInterface> getSwKernelTiledTypes(VPUIP::SwKernelOp swKernelOp, Dim tileDim) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "topk") {
        // For SW TopK, input, output and target shape will be tiled
        const auto inputType = swKernelOp->getOperand(0).getType();
        const auto auxType = swKernelOp->getOperand(1).getType();
        const auto outputType = swKernelOp->getResult(0).getType();
        const auto targetShapeType = swKernelOp->getResult(1).getType();
        return {inputType, auxType, outputType, targetShapeType};
    } else if (kernelEntryName == "gather") {
        auto args = kernelArgsRange(swKernelOp);
        const auto kernelAxisAttr = mlir::dyn_cast<mlir::IntegerAttr>(args.begin()[0]);
        VPUX_THROW_UNLESS(kernelAxisAttr != nullptr, "Failed to extract axis at '{0}'", swKernelOp->getLoc());
        const auto kernelAxis = kernelAxisAttr.getValue().getSExtValue();
        const auto axisVal = convertKernelAxisToOrigAxis(swKernelOp.getInputs()[0], kernelAxis);

        const auto inputType = swKernelOp->getOperand(0).getType();
        const auto indicesType = swKernelOp->getOperand(1).getType();
        const auto outputType = swKernelOp->getResult(0).getType();

        const auto tileDimVal = static_cast<int64_t>(tileDim.ind());

        if (tileDimVal == axisVal) {
            return {indicesType, outputType};
        }

        return {inputType, outputType};
    } else if (kernelEntryName == "gru_sequence") {
        // For SW GRUSequence, inputData, initialHiddenState and outputs will be tiled
        const auto inputDataType = swKernelOp->getOperand(0).getType();
        const auto initialHiddenStateType = swKernelOp->getOperand(1).getType();
        const auto outputYType = swKernelOp->getResult(0).getType();
        const auto outputHoType = swKernelOp->getResult(1).getType();
        return {inputDataType, initialHiddenStateType, outputYType, outputHoType};
    } else if (kernelEntryName == "gru_sequence_last_part") {
        // For SW GRUSequenceLastPart, inputData, initialHiddenState and outputs will be tiled
        const auto inputDataType = swKernelOp->getOperand(0).getType();
        const auto initialHiddenStateType = swKernelOp->getOperand(1).getType();
        const auto outputYType = swKernelOp->getResult(0).getType();
        const auto outputHoType = swKernelOp->getResult(1).getType();
        return {inputDataType, initialHiddenStateType, outputYType, outputHoType};
    } else if (kernelEntryName == "accumulate") {
        const auto lhsType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getOperand(0).getType());
        const auto rhsType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getOperand(1).getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());

        SmallVector<vpux::NDTypeInterface> tiledTypes = {lhsType, rhsType};

        const auto lhsScale = swKernelOp->getOperand(2);
        if (lhsScale != nullptr) {
            const auto lhsScaleType = mlir::cast<vpux::NDTypeInterface>(lhsScale.getType());

            // lhs Scale is broadcasted on tile axis
            if (lhsScaleType.getShape()[tileDim] != 1) {
                tiledTypes.push_back(lhsScaleType);
            }
        }

        const auto rhsScale = swKernelOp->getOperand(3);
        if (rhsScale != nullptr) {
            const auto rhsScaleType = mlir::cast<vpux::NDTypeInterface>(rhsScale.getType());

            // rhs Scale is broadcasted on tile axis
            if (rhsScaleType.getShape()[tileDim] != 1) {
                tiledTypes.push_back(rhsScaleType);
            }
        }

        tiledTypes.push_back(outputType);
        return tiledTypes;
    } else if (kernelEntryName == "eltwise_mul" || kernelEntryName == "eltwise_power" ||
               kernelEntryName == "eltwise_div" || kernelEntryName == "prelu_fp16" ||
               kernelEntryName == "eltwise_greater" || kernelEntryName == "eltwise_less" ||
               kernelEntryName == "eltwise_sub" || kernelEntryName == "eltwise_add" ||
               kernelEntryName == "eltwise_select" || kernelEntryName == "eltwise_bitwise_or" ||
               kernelEntryName == "eltwise_bitwise_and" || kernelEntryName == "eltwise_bitwise_not" ||
               kernelEntryName == "eltwise_bitwise_xor") {
        // For SW Eltwise Op with multi inputs
        // Only the input which does not need broadcast and output will be tiled
        SmallVector<vpux::NDTypeInterface> tiledTypes;
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());
        const auto outputShape = outputType.getShape();
        for (auto input : swKernelOp->getOperands()) {
            const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
            const auto inputShape = inputType.getShape();
            if (inputShape == outputShape) {
                tiledTypes.push_back(inputType);
            }
        }
        tiledTypes.push_back(outputType);

        return tiledTypes;
    } else if (kernelEntryName == "mvn6") {
        SmallVector<vpux::NDTypeInterface> tiledTypes;
        // Optional scale/bias with broadcast on 'tileDim' are not tiled
        for (auto input : swKernelOp->getOperands()) {
            const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
            const auto inputShape = inputType.getShape();
            if (inputShape[tileDim] != 1) {
                tiledTypes.push_back(inputType);
            }
        }
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());
        tiledTypes.push_back(outputType);
        return tiledTypes;
    } else if (kernelEntryName == "mvn1_norm") {
        const auto inputType = swKernelOp->getOperand(0).getType();
        const auto outputType = swKernelOp->getResult(0).getType();

        auto args = kernelArgsRange(swKernelOp);
        const auto isAcrossChannelsAttr = mlir::dyn_cast<mlir::BoolAttr>(args.begin()[0]);
        VPUX_THROW_UNLESS(isAcrossChannelsAttr != nullptr, "Failed to extract AcrossChannelsAttr at '{0}'",
                          swKernelOp->getLoc());
        const bool isAcrossChannels = isAcrossChannelsAttr.getValue();

        if (tileDim == Dims4D::Act::C && !isAcrossChannels) {
            const auto meanVarType = swKernelOp->getOperand(1).getType();
            return {inputType, meanVarType, outputType};
        }

        return {inputType, outputType};
    } else if (kernelEntryName == "rope") {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getOperand(0).getType());
        const auto cosType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getOperand(1).getType());
        const auto sinType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getOperand(2).getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(swKernelOp->getResult(0).getType());

        SmallVector<vpux::NDTypeInterface> tiledTypes = {inputType};

        if (cosType.getShape()[tileDim] == inputType.getShape()[tileDim]) {
            tiledTypes.push_back(cosType);
        }
        if (sinType.getShape()[tileDim] == inputType.getShape()[tileDim]) {
            tiledTypes.push_back(sinType);
        }

        tiledTypes.push_back(outputType);
        return tiledTypes;
    } else {
        // By default, all inputs and outputs will be tiled
        SmallVector<vpux::NDTypeInterface> tiledTypes;
        for (const auto& input : swKernelOp->getOperands()) {
            const auto inputType = input.getType();
            tiledTypes.push_back(inputType);
        }
        for (const auto& output : swKernelOp->getResults()) {
            const auto outputType = output.getType();
            tiledTypes.push_back(outputType);
        }
        return tiledTypes;
    }
}

bool isCacheOpTaskType(mlir::SymbolRefAttr kernelTaskType, bool includePrefetch) {
    if (!kernelTaskType) {
        return false;
    }
    auto taskTypeVal = VPU::symbolizeActShaveTaskType(kernelTaskType.getLeafReference().strref());
    VPUX_THROW_UNLESS(taskTypeVal.has_value(), "VPU::ActShaveTaskType has no value.");
    if (taskTypeVal.value() == VPU::ActShaveTaskType::COMPUTE) {
        return false;
    }

    if (!includePrefetch && taskTypeVal.value() == VPU::ActShaveTaskType::CACHE_PREFETCH) {
        return false;
    }
    return true;
}

bool isCacheOpTaskType(std::optional<::mlir::SymbolRefAttr> kernelTaskType, bool includePrefetch) {
    return kernelTaskType.has_value() ? isCacheOpTaskType(kernelTaskType.value(), includePrefetch) : false;
}

bool isCacheHandlingOp(VPUIP::SwKernelOp swKernelOp) {
    auto moduleOp = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(swKernelOp.getKernelFunctionAttr());
    if (kernelFunc == nullptr) {
        return false;
    }

    auto kernelTaskType = kernelFunc->getAttrOfType<mlir::SymbolRefAttr>("VPU.task_type");
    if (kernelTaskType == nullptr) {
        return false;
    }

    return isCacheOpTaskType(kernelTaskType);
}

mlir::SmallVector<mlir::Value> getDDRBuffers(mlir::ValueRange buffers) {
    mlir::SmallVector<mlir::Value> ddrBuffers;
    llvm::copy(buffers | vpux::filtered([](mlir::Value buffer) {
                   auto bufferType = mlir::cast<vpux::NDTypeInterface>(buffer.getType());
                   return bufferType.getMemoryKind() == VPU::MemoryKind::DDR;
               }),
               std::back_inserter(ddrBuffers));

    return ddrBuffers;
}

bool hasInputsInDDR(VPUIP::SwKernelOp swKernelTask) {
    return llvm::any_of(swKernelTask.getInputs(), [](mlir::Value buffer) {
        auto bufferType = mlir::cast<vpux::NDTypeInterface>(buffer.getType());
        if (bufferType.getMemoryKind() == VPU::MemoryKind::DDR) {
            return true;
        }
        return false;
    });
}

int64_t getSwKernelTilingAddressAlignment(VPUIP::SwKernelOp swkernelOp, config::ArchKind arch) {
    if (arch == config::ArchKind::NPU37XX) {
        return 1;
    }

    auto name = getSwKernelEntryName(swkernelOp);
    if (llvm::find(SW_KERNELS_NEED_TILING_ALIGNMENT, name) == SW_KERNELS_NEED_TILING_ALIGNMENT.end()) {
        return 1;
    }
    return NPU40XX_SW_KERNEL_ADDRESS_ALIGNMENT;
}

std::pair<bool, size_t> getSwKernelInstructionPrefetchConfig(config::ArchKind arch) {
    // Return {useDummyKernelForInstructionPrefetch, minimumShaveStartTimeForPrefetch}
    switch (arch) {
    case config::ArchKind::NPU40XX:
        return std::make_pair(true, MIN_FREE_CYCLES_FOR_PREFETCH_280K);
    case config::ArchKind::NPU50XX:
        return std::make_pair(false, MIN_FREE_CYCLES_FOR_PREFETCH_250K);
    default:
        VPUX_THROW("Unsupported Arch {0} to do Shave Instruction Prefetch", arch);
    }
}

}  // namespace VPUIP
}  // namespace vpux
