//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/core/act_profiling.hpp"

#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <algorithm>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ACTSHAVEPROFILING
#define GEN_PASS_DEF_ACTSHAVEPROFILING
#define GEN_PASS_DECL_GROUPPROFILINGBUFFERS
#define GEN_PASS_DEF_GROUPPROFILINGBUFFERS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// ActShaveProfilingPass
//

class ActShaveProfilingPass final : public VPUIP::impl::ActShaveProfilingBase<ActShaveProfilingPass> {
public:
    explicit ActShaveProfilingPass(VPUIP::MemKindCreateFunc memKindCb, Logger log): _memKindCb(std::move(memKindCb)) {
        VPUX_THROW_UNLESS(_memKindCb != nullptr, "Missing memKindCb");
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    VPUIP::MemKindCreateFunc _memKindCb;
};

void ActShaveProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    auto maybeMemKind = _memKindCb("");
    if (!maybeMemKind.has_value()) {
        _log.trace("Memory Space is not defined");
        return;
    }

    vpux::IndexedSymbolAttr memKindAttr = nullptr;
    {
        const auto memKind = maybeMemKind.value();
        if (memKind == VPU::MemoryKind::CMX_NN) {
            memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(memKind), 0);
        } else {
            memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(memKind));
        }
    }

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp::getFromModule(module, netInfo, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    SmallVector<VPUIP::SwKernelOp> swTasks;
    bool isClusteredActShavePresent = false;
    netFunc.walk([&](VPUIP::SwKernelOp swKernelOp) {
        _log.trace("Process Operation '{0}'", swKernelOp->getLoc());

        for (const auto& resultType : swKernelOp->getResultTypes()) {
            isClusteredActShavePresent |= (mlir::dyn_cast<VPUIP::DistributedBufferType>(resultType) != nullptr);
        }
        swTasks.push_back(swKernelOp);
    });

    // No ActShave tasks in the network, nothing to profile
    if (swTasks.empty()) {
        return;
    }

    const auto tileCount = static_cast<unsigned>(config::getTileExecutor(module).getCount());
    ActShaveProfiler profiler(isClusteredActShavePresent ? tileCount : 1, builder, ctx, memKindAttr, netFunc, _log,
                              std::make_shared<NameUniqifier>(_log));

    for (auto& swTask : swTasks) {
        profiler.scheduleTask(swTask);
    }

    // Declare and create additional output from network
    const unsigned outputDdrSize = profiler.getRequiredDdrMemory();
    const auto outputResultDdr = mlir::MemRefType::get({outputDdrSize}, getUInt32Type(ctx));
    auto profilingResult =
            addNewProfilingOutput(ctx, netFunc, netInfo, outputResultDdr, profiling::ExecutorType::ACTSHAVE);

    SmallVector<mlir::Value> concatResults;
    profiler.addProfilingOps(profilingResult, concatResults);

    mlir::func::ReturnOp returnOp =
            mlir::dyn_cast_or_null<mlir::func::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);

    auto concatview = builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(ctx, "actshaveDDRProfiling")), concatResults, profilingResult);
    returnOp.getOperandsMutable().append(concatview.getOutput());

    // After profiling was added and ActShave tasks were recreated with profiling outputs added
    // remove old operations
    for (auto& swTask : swTasks) {
        swTask.erase();
    }
    ActShaveProfiler::resetBufferIdCounter();
}

//
// GroupProfilingBuffersPass
//

class GroupProfilingBuffersPass final : public VPUIP::impl::GroupProfilingBuffersBase<GroupProfilingBuffersPass> {
public:
    explicit GroupProfilingBuffersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    static unsigned getAlignment(StringRef name);
    void safeRunOnModule() final;
};

void GroupProfilingBuffersPass::safeRunOnModule() {
    auto ctx = &getContext();
    auto module = getOperation();

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp::getFromModule(module, netInfo, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    // If profiling was enabled for SW kernels only and network doesn't have any, there will be
    // getProfilingOutputsInfo with single block "^bb0" and 0 operation inside.
    if (netInfo.getProfilingOutputsInfo().empty() || netInfo.getProfilingOutputsInfo().front().empty() ||
        netInfo.getProfilingOutputsInfo().front().getOps().empty()) {
        return;
    }

    // Collecting sizes of all profiling buffers in order to calculate offsets in the base buffer
    // New buffer name will be like: [offset]_[name]_[offset]_[name]...
    auto& profilingOutputs = netInfo.getProfilingOutputsInfo().front();
    SmallVector<uint32_t> outputBases;

    struct Section {
        int execType;
        size_t offset;
        size_t size;
    };
    std::vector<Section> sections;
    size_t offset = 0;
    profilingOutputs.walk([&](net::DataInfoOp op) {
        const auto type = mlir::cast<vpux::NDTypeInterface>(op.getUserType());
        const auto sectionName = op.getName();

        offset = alignValUp(offset, profiling::PROFILING_SECTION_ALIGNMENT);
        const auto size = static_cast<size_t>(type.getCompactAllocSize().count());

        const auto execType = profiling::convertDataInfoNameToExecType(sectionName);
        const auto execCode = static_cast<int>(execType);
        sections.push_back({execCode, offset, size});
        outputBases.push_back(checked_cast<unsigned int>(offset));
        offset += size;
        op.erase();
    });
    const size_t totalSize = alignValUp(offset, profiling::PROFILING_SECTION_ALIGNMENT);

    //
    // Create and add new combined profiling output to the user info
    //
    auto newOutputResult =
            mlir::MemRefType::get({static_cast<int64_t>(totalSize / sizeof(uint32_t))}, getUInt32Type(ctx));
    auto newOutputShapedType = mlir::cast<vpux::NDTypeInterface>(newOutputResult);
    auto outputUserResult = getTensorType(newOutputShapedType.getShape(), newOutputShapedType.getElementType(),
                                          newOutputShapedType.getDimsOrder(), nullptr);
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&profilingOutputs.front(), &builderLog);

    auto dataInfo = userInfoBuilder.create<net::DataInfoOp>(
            mlir::UnknownLoc::get(ctx), mlir::StringAttr::get(ctx, profiling::PROFILING_OUTPUT_NAME),
            mlir::TypeAttr::get(outputUserResult),
            /*OptionalAttr originalShape*/ nullptr,
            /*OptionalAttr friendlyName*/ nullptr,
            /*OptionalAttr inputName*/ nullptr,
            /*OptionalAttr tensorNames*/ nullptr,
            /*profilingSectionsCount=*/1);
    dataInfo.getSections().front().emplaceBlock();

    auto sectionsBuilder =
            mlir::OpBuilder::atBlockBegin(&dataInfo.getSections().front().front(), builder.getListener());
    for (const auto& section : sections) {
        sectionsBuilder.create<VPUIP::ProfilingSectionOp>(mlir::UnknownLoc::get(ctx), getIntAttr(ctx, section.execType),
                                                          getIntAttr(ctx, section.offset),
                                                          getIntAttr(ctx, section.size));
    }

    auto totalArgumentsCount = netFunc.getNumArguments();
    auto mainArgumentsCount = totalArgumentsCount - outputBases.size();
    auto numInputs = netFunc.getNumArguments() - netFunc.getNumResults();
    auto numMainOutputs = mainArgumentsCount - numInputs;
    VPUX_THROW_UNLESS(mainArgumentsCount > 0, "There is no main network arguments in the funcOp");

    // Adding new output buffer
    const mlir::Location suffixLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "profiling_result"));
    const auto profilingResultLoc = mlir::FusedLoc::get(ctx, {netFunc.getLoc(), suffixLoc});
    netFunc.insertArgument(totalArgumentsCount, newOutputResult, nullptr, profilingResultLoc);
    auto newProfilngResult = netFunc.getArgument(totalArgumentsCount);
    SmallVector<unsigned> argsToErase;
    unsigned removedArgs = 0;
    // Loop thought the old function arguments in order to replace its usages
    // with DeclareBufferOp with previously calculated offset connected to the new buffer
    // and erase these arguments
    auto arg = netFunc.getArguments().begin();
    while (arg != netFunc.getArguments().end()) {
        auto argNum = arg->getArgNumber();
        if (argNum > mainArgumentsCount - 1) {
            while (!arg->use_empty()) {
                auto use = arg->use_begin();
                auto op = use->getOwner();
                auto taskOp = op->getParentOfType<VPURT::TaskOp>();
                builder.setInsertionPoint((taskOp != nullptr) ? taskOp : op);
                unsigned base = (removedArgs) ? outputBases[removedArgs] : 0;
                auto declareOp = builder.create<VPURT::DeclareBufferOp>(
                        mlir::NameLoc::get(mlir::StringAttr::get(ctx, "newProfilingBuffer")), arg->getType(),
                        VPURT::BufferSection::ProfilingOutput, 0, base);
                use->set(declareOp.getBuffer());
            }
            netFunc.eraseArgument(argNum);
            removedArgs++;
            // Reach the last old profiling output, stop here as the next output is the new buffer
            if (removedArgs == outputBases.size()) {
                break;
            }
        } else {
            arg++;
        }
    }

    //
    // Recalculate existing DeclareTensorOp
    //
    netFunc.walk([&](VPURT::DeclareBufferOp op) {
        if (op.getSection() == VPURT::BufferSection::ProfilingOutput ||
            op.getSection() == VPURT::BufferSection::NetworkOutput) {
            auto sectionIndex = op.getSectionIndex();
            if (sectionIndex.has_value()) {
                VPUX_THROW_UNLESS(sectionIndex.value().size() == 1,
                                  "Profiling output is expected to have just one locale index");
                auto idx = parseIntArrayAttr<int64_t>(sectionIndex.value())[0];
                if (op.getSection() == VPURT::BufferSection::NetworkOutput) {
                    if (idx < static_cast<int64_t>(numMainOutputs)) {
                        return;
                    }
                    idx -= numMainOutputs;
                    auto sectionAttr = VPURT::BufferSectionAttr::get(ctx, VPURT::BufferSection::ProfilingOutput);
                    op.setSectionAttr(sectionAttr);
                } else if (idx <= 0) {
                    return;
                }

                auto base = outputBases[idx];
                op.setSectionIndexAttr(getIntArrayAttr(ctx, SmallVector<int64_t>({0})));
                auto offset = static_cast<uint32_t>(base + op.getByteOffset());
                op.setByteOffsetAttr(builder.getUI32IntegerAttr(offset));
            }
        }
    });

    //
    // Replace function signature
    //
    auto funcType = netFunc.getFunctionType();
    auto newResultTypes = to_small_vector(llvm::concat<const mlir::Type>(
            funcType.getResults().drop_back(outputBases.size()), ArrayRef(newOutputResult)));
    auto newFunctionType = mlir::FunctionType::get(ctx, funcType.getInputs(), newResultTypes);
    netFunc.setType(newFunctionType);

    //
    // Replace function return operands
    //
    netFunc.walk([&](mlir::func::ReturnOp op) {
        auto start = static_cast<unsigned>(op.getOperandsMutable().size() - outputBases.size());
        op.getOperandsMutable().erase(start, static_cast<unsigned>(outputBases.size()));
        op.getOperandsMutable().append(newProfilngResult);
    });
}

}  // namespace

//
// createActShaveProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createActShaveProfilingPass(MemKindCreateFunc memKindCb, Logger log) {
    return std::make_unique<ActShaveProfilingPass>(std::move(memKindCb), log);
}

//
// createGroupProfilingBuffersPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createGroupProfilingBuffersPass(Logger log) {
    return std::make_unique<GroupProfilingBuffersPass>(log);
}
