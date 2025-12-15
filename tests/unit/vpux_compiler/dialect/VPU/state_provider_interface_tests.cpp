//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common/utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/state_provider_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_opt_alg.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_state_provider.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>
#include <iostream>
#include <random>

#include "llvm/Bitcode/BitcodeReader.h"

using vpux::config::ArchKind;
using namespace vpux;

std::mt19937 gen(1);

class StateProviderImpl : public VPU::IStateProvider {
private:
    std::shared_ptr<OperationStrategies> _storage;

public:
    StateProviderImpl(std::shared_ptr<OperationStrategies> strategies): _storage(strategies) {
    }
    VPU::OperationStrategy getState(int /*temperature*/, double& /*cost*/,
                                    const VPU::OperationStrategy* const state) override {
        if (state == nullptr) {
            const auto allOperations = _storage->getAllOperations();
            VPUX_THROW_WHEN(allOperations.empty(), "There are no operations added in this state");

            std::uniform_int_distribution<> opDistribution(0, allOperations.size() - 1);
            mlir::Operation* chosenOp = allOperations[opDistribution(gen)];

            VPUX_THROW_WHEN((chosenOp == nullptr || !_storage->hasAnyStrategy(chosenOp)),
                            "Invalid op. There are no strategies added for this op");
            auto currentStrategy = _storage->getCurrentStrategy(chosenOp);

            return std::make_pair(chosenOp, currentStrategy);
        }

        mlir::Operation* stateOp = state->first;
        VPUX_THROW_WHEN(!_storage->hasAnyStrategy(stateOp), "Invalid op. There are no strategies added for this op");

        const auto allStrategies = _storage->getAllStrategies(stateOp);
        auto chosenStrategy = allStrategies[0].strategy;
        if (allStrategies.size() == 1) {
            return std::make_pair(stateOp, chosenStrategy);
        } else {
            do {
                std::uniform_int_distribution<> strategyDistribution(0, allStrategies.size() - 1);
                chosenStrategy = allStrategies[strategyDistribution(gen)].strategy;
            } while (chosenStrategy == state->second);
        }

        return std::make_pair(stateOp, chosenStrategy);
    }
    void updateState(const VPU::OperationStrategy& state) override {
        _storage->setCurrentStrategy(state);
    }

    VPU::StrategyCost getCost(const VPU::OperationStrategy& state) override {
        return _storage->getStrategyCost(state);
    }

    VPU::StrategyCost getFullCost() override {
        const auto allOperations = _storage->getAllOperations();

        VPUX_THROW_WHEN(allOperations.empty(), "There are no operations added in this state");

        auto totalcost = std::accumulate(
                allOperations.begin(), allOperations.end(), 0, [&](VPU::StrategyCost total, mlir::Operation* chosenOp) {
                    VPU::Strategy chosenStrategy = _storage->getCurrentStrategy(chosenOp);
                    return total + _storage->getStrategyCost(std::make_pair(chosenOp, chosenStrategy));
                });
        return totalcost;
    }

    void updateSolution(const VPU::OperationStrategy& state) override {
        for (auto* operation : _storage->getAllOperations()) {
            _storage->setBestStrategy(std::make_pair(operation, _storage->getCurrentStrategy(operation)));
        }
        _storage->setBestStrategy(state);
    }
};

using StateProviderInterfaceTests = vpux::VPU::arch37xx::UnitTest;

TEST_F(StateProviderInterfaceTests, StateProvider_tests) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
        %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
        %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
            { ppe = #VPU.PPEStub<>, pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>,
            rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
            : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x4xsi32> -> tensor<1x80x28x28xf16, {order = #NHWC}> loc(fused["Conv_100", "t_Convolution"])
        %1 = VPU.Tanh(%0) : tensor<1x80x28x28xf16, {order = #NHWC}>
            -> tensor<1x80x28x28xf16, {order = #NHWC}> loc(fused["Tanh_1", "t_Convolution"])
        %2 = VPU.NCE.MaxPool(%1, %cst)
            {kernel_size = [1, 1], ppe = #VPU.PPEStub<>,
            pad =  #VPU.Padding<bottom = 0, left = 0, right = 0, top = 0>, strides = [1, 1]}
            -> tensor<1x80x28x28xf16, {order = #NHWC}> loc(fused["Maxpool_1", "fused","t_Convolution"])
        return %2 : tensor<1x80x28x28xf16, {order = #NHWC}>
    }
  }
)";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(ArchKind::NPU37XX, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto storage = std::make_shared<OperationStrategies>();

    VPU::Strategy splitOverHStrategy(VPU::MultiClusterStrategy::SplitOverHeight, nullptr);
    VPU::Strategy splitOverKStrategy(VPU::MultiClusterStrategy::SplitOverKernel,
                                     getIntArrayAttr(&ctx, ArrayRef({1, 1, 2, 1})));
    VPU::Strategy multiClusteringStrategy(VPU::MultiClusterStrategy::Clustering, nullptr);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto convSOH = std::make_pair(convOp, splitOverHStrategy);
        auto convSOK = std::make_pair(convOp, splitOverKStrategy);
        auto convClst = std::make_pair(convOp, multiClusteringStrategy);

        storage->addStrategy(convSOH, 1000);
        storage->addStrategy(convSOK, 1200);
        storage->addStrategy(convClst, 1400);

        storage->setCurrentStrategy(convSOK);
        storage->setBestStrategy(convSOK);

        EXPECT_EQ(storage->getCurrentStrategy(convOp), splitOverKStrategy);
    });

    func->walk([&](VPU::TanhOp tanhOp) {
        auto tanhSOH = std::make_pair(tanhOp, splitOverHStrategy);
        auto tanhSOK = std::make_pair(tanhOp, splitOverKStrategy);
        auto tanhClst = std::make_pair(tanhOp, multiClusteringStrategy);

        storage->addStrategy(tanhSOH, 2000);
        storage->addStrategy(tanhSOK, 2200);
        storage->addStrategy(tanhClst, 2400);

        storage->setCurrentStrategy(tanhSOK);
        storage->setBestStrategy(tanhSOK);
        EXPECT_EQ(storage->getCurrentStrategy(tanhOp), splitOverKStrategy);
    });

    func->walk([&](VPU::NCEMaxPoolOp maxOp) {
        auto maxSOH = std::make_pair(maxOp, splitOverHStrategy);
        auto maxSOK = std::make_pair(maxOp, splitOverKStrategy);
        auto maxClst = std::make_pair(maxOp, multiClusteringStrategy);

        storage->addStrategy(maxSOH, 3000);
        storage->addStrategy(maxSOK, 3200);
        storage->addStrategy(maxClst, 3400);

        storage->setCurrentStrategy(maxClst);
        storage->setBestStrategy(maxClst);
    });

    auto stateProvider = std::make_shared<StateProviderImpl>(storage);
    auto testAlgorithm = createAlgorithm(vpux::VPU::TilingOptions{}, stateProvider, storage);
    testAlgorithm->optimize();

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        EXPECT_EQ(splitOverHStrategy, storage->getBestStrategy(convOp));
    });

    func->walk([&](VPU::TanhOp tanhOp) {
        EXPECT_EQ(splitOverHStrategy, storage->getBestStrategy(tanhOp));
    });

    func->walk([&](VPU::NCEMaxPoolOp maxOp) {
        EXPECT_EQ(splitOverHStrategy, storage->getBestStrategy(maxOp));
    });
}

TEST_F(StateProviderInterfaceTests, DefaultStateProvider_tests) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
        %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
        %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
            {pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>,
            ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
            : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x4xsi32> -> tensor<1x80x28x28xf16, {order = #NHWC}> loc(fused["Conv_100", "t_Convolution"])
        %1 = VPU.Tanh(%0) : tensor<1x80x28x28xf16, {order = #NHWC}>
            -> tensor<1x80x28x28xf16, {order = #NHWC}> loc(fused["Tanh_1", "t_Convolution"])
        %2 = VPU.NCE.MaxPool(%1, %cst)
            {kernel_size = [1, 1], ppe = #VPU.PPEStub<>,
            pad =  #VPU.Padding<bottom = 0, left = 0, right = 0, top = 0>, strides = [1, 1]}
            -> tensor<1x80x28x28xf16, {order = #NHWC}> loc(fused["Maxpool_1", "fused","t_Convolution"])
        return %2 : tensor<1x80x28x28xf16, {order = #NHWC}>
    }
  }
)";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(ArchKind::NPU37XX, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto storage = std::make_shared<OperationStrategies>();

    VPU::Strategy splitOverHStrategy(VPU::MultiClusterStrategy::SplitOverHeight, nullptr);
    VPU::Strategy splitOverKStrategy(VPU::MultiClusterStrategy::SplitOverKernel,
                                     getIntArrayAttr(&ctx, ArrayRef({1, 1, 2, 1})));
    VPU::Strategy multiClusteringStrategy(VPU::MultiClusterStrategy::Clustering, nullptr);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto convSOH = std::make_pair(convOp, splitOverHStrategy);
        auto convSOK = std::make_pair(convOp, splitOverKStrategy);
        auto convClst = std::make_pair(convOp, multiClusteringStrategy);

        storage->addStrategy(convSOH, 1000);
        storage->addStrategy(convSOK, 1200);
        storage->addStrategy(convClst, 1400);

        storage->setCurrentStrategy(convSOK);
        storage->setBestStrategy(convSOK);
    });

    func->walk([&](VPU::TanhOp tanhOp) {
        auto tanhSOH = std::make_pair(tanhOp, splitOverHStrategy);
        auto tanhSOK = std::make_pair(tanhOp, splitOverKStrategy);

        storage->addStrategy(tanhSOH, 2000);
        storage->addStrategy(tanhSOK, 2200);

        storage->setCurrentStrategy(tanhSOK);
        storage->setBestStrategy(tanhSOK);
    });

    func->walk([&](VPU::NCEMaxPoolOp maxOp) {
        auto maxSOH = std::make_pair(maxOp, splitOverHStrategy);
        auto maxSOK = std::make_pair(maxOp, splitOverKStrategy);
        auto maxClst = std::make_pair(maxOp, multiClusteringStrategy);

        storage->addStrategy(maxSOH, 3000);
        storage->addStrategy(maxSOK, 3200);
        storage->addStrategy(maxClst, 3400);

        storage->setCurrentStrategy(maxClst);
        storage->setBestStrategy(maxClst);
    });
    const auto vpunnCostFunc = std::make_shared<LayerVPUNNCost>(func);
    const auto stateProvider = std::make_shared<DefaultStateProvider>(storage, vpunnCostFunc);
    auto testAlgorithm = createAlgorithm(vpux::VPU::TilingOptions{}, stateProvider, storage);
    testAlgorithm->optimize();

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        EXPECT_EQ(splitOverHStrategy, storage->getBestStrategy(convOp));
    });

    func->walk([&](VPU::TanhOp tanhOp) {
        EXPECT_EQ(splitOverHStrategy, storage->getBestStrategy(tanhOp));
    });

    func->walk([&](VPU::NCEMaxPoolOp maxOp) {
        EXPECT_EQ(splitOverHStrategy, storage->getBestStrategy(maxOp));
    });
}

TEST_F(StateProviderInterfaceTests, StateProviderNCEPermute_tests) {
    ctx.loadDialect<IE::IEDialect>();

    constexpr llvm::StringLiteral inputIR = R"(
!qElemType0 = !quant.uniform<u8:f16, 0.92406077665441178>
!qElemType1 = !quant.uniform<u8<0:254>:f16:0, {0.0035545640573726865:127,0.0042422878460621274:127,0.0049146836198221038:127,6.0496315008073346E-4:127,0.004828331508035735:127,0.001181764161492896:127,0.005209280749944251:127,8.2280633487100672E-4:127,2.8928686080016491E-4:127,0.0010465964322953713:127,0.002631273091308714:127,0.0080264891226460612:127,0.0041658934645765408:127,0.0041675689652210142:127,0.0038724478304855469:127,0.003833929150123296:127,0.0071884792620741473:127,0.0044150849965613661:127,0.0047772418795608163:127,0.0070094231545455811:127,0.0033757365125370777:127,0.0031073365624495379:127,0.0059386271191394233:127,0.0047132203898091951:127,0.0085115019730695583:127,5.3644256563637198E-4:127,0.0047396703029242088:127,0.0042686959889930067:127,0.0072794922693507876:127,5.1457282361083148E-4:127,0.0032727173932894007:127,0.010880531288507416:127}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main attributes {config.arch = #config.arch_kind<NPU37XX>} {
        func.func @main(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x32x112x112x!qElemType0, {order = #NHWC}> {
        %cst = const.Declare tensor<32x1x1x32x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x1x1x32xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
        %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<0> : tensor<32x1x1x4xsi32>
        %cst_1 = const.Declare tensor<32x16x1x1x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]

        %0 = VPU.NCE.Permute(%arg0)
                {dstElemType = !qElemType0, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEStub<>}
                -> tensor<1x4x224x224x!qElemType0, {order = #NHWC}>
        %1 = VPU.NCE.CompressConvolution(%0, %cst, %cst_0)
               {cm_sp_pattern = 7 : i64, ppe = #VPU.PPEStub<>,
               pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
               rawFilterShape = [32, 3, 3, 3], strides = [2, 2]}
               -> tensor<1x32x112x112x!qElemType0, {order = #NHWC}>
        %2 = VPU.NCE.DepthConvolution(%1, %cst_1, %cst_0)
               {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
                -> tensor<1x32x112x112x!qElemType0, {order = #NHWC}>
        return %2 : tensor<1x32x112x112x!qElemType0, {order = #NHWC}>
    }
  }
)";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    module.get()->removeAttr("config.arch");
    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(ArchKind::NPU37XX, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto storage = std::make_shared<OperationStrategies>();

    VPU::Strategy splitOverHStrategy(VPU::MultiClusterStrategy::SplitOverHeight, nullptr);
    VPU::Strategy splitOverHOverStrategy(VPU::MultiClusterStrategy::SplitOverHeightOverlapped, nullptr);
    VPU::Strategy splitOverKStrategy(VPU::MultiClusterStrategy::SplitOverKernel, nullptr);
    VPU::Strategy multiClusteringStrategy(VPU::MultiClusterStrategy::Clustering, nullptr);

    func->walk([&](VPU::NCECompressConvolutionOp convOp) {
        auto convSOHOver = std::make_pair(convOp, splitOverHOverStrategy);
        auto convClst = std::make_pair(convOp, multiClusteringStrategy);

        storage->addStrategy(convSOHOver, 12929);
        storage->addStrategy(convClst, 42762);

        storage->setCurrentStrategy(convClst);
        storage->setBestStrategy(convClst);
    });
    func->walk([&](VPU::NCEDepthConvolutionOp depthOp) {
        auto depthSOH = std::make_pair(depthOp, splitOverHStrategy);
        auto depthSOK = std::make_pair(depthOp, splitOverKStrategy);
        auto depthClst = std::make_pair(depthOp, multiClusteringStrategy);

        storage->addStrategy(depthSOH, 12017);
        storage->addStrategy(depthSOK, 40830);
        storage->addStrategy(depthClst, 38669);

        storage->setCurrentStrategy(depthSOK);
        storage->setBestStrategy(depthSOK);
    });
    func->walk([&](VPU::NCEPermuteOp permuteOp) {
        auto permuteSOHOver = std::make_pair(permuteOp, splitOverHOverStrategy);

        storage->addStrategy(permuteSOHOver, 100352);

        storage->setCurrentStrategy(permuteSOHOver);
        storage->setBestStrategy(permuteSOHOver);
    });

    const auto vpunnCostFunc = std::make_shared<LayerVPUNNCost>(func);
    const auto stateProvider = std::make_shared<DefaultStateProvider>(storage, vpunnCostFunc);
    auto testAlgorithm = createAlgorithm(vpux::VPU::TilingOptions{}, stateProvider, storage);
    testAlgorithm->optimize();

    func->walk([&](VPU::NCECompressConvolutionOp convOp) {
        EXPECT_EQ(splitOverHOverStrategy, storage->getBestStrategy(convOp));
    });

    func->walk([&](VPU::NCEDepthConvolutionOp tanhOp) {
        EXPECT_EQ(splitOverHStrategy, storage->getBestStrategy(tanhOp));
    });

    func->walk([&](VPU::NCEPermuteOp pqOp) {
        EXPECT_EQ(splitOverHOverStrategy, storage->getBestStrategy(pqOp));
    });
}

TEST_F(StateProviderInterfaceTests, StateProviderMultiUsers_tests) {
    constexpr llvm::StringLiteral inputIR = R"(
!qElemType0 = !quant.uniform<u8:f16, 0.92406077665441178>
!qElemType1 = !quant.uniform<u8<0:254>:f16:0, {0.0017211093442646536:127,9.2923594272042824E-4:127,0.0018374221766088892:127,0.0025908069347772072:127,0.0017101335243915948:127,0.0018683784589992735:127,0.0016507278746507299:127,0.0017394461031035177:127,0.0018071128627446694:127,0.0011389537120428611:127,0.0015641700799070943:127,0.0012958705659926407:127,0.0010234508223420991:127,0.0020769322951008956:127,0.0017929582849262268:127,0.0011920894928804534:127,0.0013518602125287995:127,0.0012784321007766122:127,0.001515371710296691:127,0.0015409595853700412:127,0.0016217116765149934:127,0.0023230938460883192:127,0.0018195782120772235:127,0.0019883576809890625:127,0.0014683198271773933:127,0.0018947339198720737:127,0.0015469348336767962:127,0.0014583249026396143:127,0.0021757190152416079:127,0.0017517461316792044:127,0.0013403609747023094:127,0.0019222899215427908:127,0.0013983010307071716:127,0.0020226594500654324:127,0.0016001387843935508:127,0.0016631281047355471:127,0.0018111743091598271:127,0.0016225850957585133:127,0.0016542500632954395:127,0.0010099353518073014:127,0.0018822333709461482:127,0.0014747883391192579:127,0.001074063378994859:127,0.0016912785102063278:127,0.0013028490027104775:127,0.0011288510767493661:127,0.0011626575875470018:127,0.0013496842909985641:127,0.0012036693377757636:127,0.0016874746074826699:127,0.0013355735953398578:127,0.0018489633019514911:127,0.0010105796216979741:127,0.0016439619261448778:127,0.0011915377979203471:127,0.0020198699996227354:127,9.2219725603193743E-4:127,0.0015577398181900264:127,0.0014870618506679385:127,0.0020814814905482015:127,0.0013138790299573283:127,0.0018591194406269104:127,0.0013782542756223303:127,0.0017069824567929966:127}>
!qElemType2 = !quant.uniform<u8<0:254>:f16:0, {0.0010324811606895266:127,8.8973912432437812E-4:127,0.0016594099716877374:127,0.0010915999337444155:127,0.0011557027345567238:127,8.92391531016883E-4:127,0.0011023545828391249:127,8.6214170446546054E-4:127,0.0013064665822532233:127,0.0011416598567812462:127,0.0012382421437210924:127,0.0010337116211418091:127,6.9184982635843472E-4:127,9.6301270986166528E-4:127,7.3782383926271453E-4:127,8.3580789134258355E-4:127,0.0010083674445865662:127,8.3392752906468913E-4:127,8.9091086012171947E-4:127,0.0012715925851206141:127,6.7179192473569257E-4:127,0.001004067462260329:127,7.3549761546878368E-4:127,8.4082101743052323E-4:127,9.5527777521629029E-4:127,9.0008041286093038E-4:127,0.0010675579074799545:127,7.210945871871287E-4:127,0.001423991805925144:127,7.5881341545600586E-4:127,0.001017582580799193:127,8.087868061591321E-4:127}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x48x7x7x!qElemType0, {order = #NHWC}> ) -> tensor<1x128x7x7x!qElemType0, {order = #NHWC}> {
        %cst = const.Declare tensor<64x48x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<64x48x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
        %cst_0 = const.Declare tensor<64x1x1x4xsi32> = dense<0> : tensor<64x1x1x4xsi32>
        %cst_1 = const.Declare tensor<32x64x3x3x!qElemType2, {order = #NHWC}> = dense<1.0> : tensor<32x64x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
        %cst_2 = const.Declare tensor<32x1x1x4xsi32> = dense<0> : tensor<32x1x1x4xsi32>
        %cst_3 = const.Declare tensor<32x96x3x3x!qElemType2, {order = #NHWC}> = dense<1.0> : tensor<32x96x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]

        %0 = VPU.NCE.Convolution(%arg0, %cst, %cst_0)
           {ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [64, 48, 3, 3], strides = [1, 1]}
            : tensor<1x48x7x7x!qElemType0, {order = #NHWC}>, tensor<64x48x3x3x!qElemType1, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x7x7x!qElemType0, {order = #NHWC}>
        %1 = VPU.NCE.Convolution(%0, %cst_1, %cst_2)
           {ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [32, 64, 3, 3], strides = [1, 1]}
            : tensor<1x64x7x7x!qElemType0, {order = #NHWC}>, tensor<32x64x3x3x!qElemType2, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x7x7x!qElemType0, {order = #NHWC}>
        %2 = VPU.Concat(%0, %1)
           {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]}
           : tensor<1x64x7x7x!qElemType0, {order = #NHWC}>, tensor<1x32x7x7x!qElemType0, {order = #NHWC}>
           -> tensor<1x96x7x7x!qElemType0, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_3, %cst_2)
           {ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            rawFilterShape = [32, 96, 3, 3], strides = [1, 1]} : tensor<1x96x7x7x!qElemType0, {order = #NHWC}>, tensor<32x96x3x3x!qElemType2, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x7x7x!qElemType0, {order = #NHWC}>
        %4 = VPU.Concat(%0, %1, %3) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}
           : tensor<1x64x7x7x!qElemType0, {order = #NHWC}>, tensor<1x32x7x7x!qElemType0, {order = #NHWC}>,
             tensor<1x32x7x7x!qElemType0, {order = #NHWC}> -> tensor<1x128x7x7x!qElemType0, {order = #NHWC}>
        return %4 : tensor<1x128x7x7x!qElemType0, {order = #NHWC}>
    }
  }
)";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(ArchKind::NPU37XX, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto storage = std::make_shared<OperationStrategies>();
    const auto vpunnCostFunc = std::make_shared<LayerVPUNNCost>(func);

    VPU::Strategy splitOverHStrategy(VPU::MultiClusterStrategy::SplitOverHeight, nullptr);
    VPU::Strategy HKSwitchStrategy(VPU::MultiClusterStrategy::HKSwitch, nullptr);
    VPU::Strategy splitOverKStrategy(VPU::MultiClusterStrategy::SplitOverKernel, nullptr);
    VPU::Strategy multiClusteringStrategy(VPU::MultiClusterStrategy::Clustering, nullptr);

    func->walk([&](VPU::ConcatOp concatOp) {
        auto concatSOH = std::make_pair(concatOp, splitOverHStrategy);
        auto concatClst = std::make_pair(concatOp, multiClusteringStrategy);
        auto concatSOK = std::make_pair(concatOp, splitOverKStrategy);
        auto concatHK = std::make_pair(concatOp, HKSwitchStrategy);

        storage->addStrategy(concatSOH, 0);
        storage->addStrategy(concatClst, 0);
        storage->addStrategy(concatSOK, 0);
        storage->addStrategy(concatHK, 0);

        storage->setCurrentStrategy(concatSOH);
        storage->setBestStrategy(concatSOH);
    });
    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto convSOH = std::make_pair(convOp, splitOverHStrategy);
        auto convSOK = std::make_pair(convOp, splitOverKStrategy);
        auto convClst = std::make_pair(convOp, multiClusteringStrategy);
        auto convHK = std::make_pair(convOp, HKSwitchStrategy);

        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(convOp.getOperation());
        const auto addStrategy = [&](const VPU::OperationStrategy& opStrategy) {
            if (clusteredOp != nullptr) {
                clusteredOp.setMultiClusterStrategy(opStrategy.second.getMCStrategy());
            }
            storage->addStrategy(opStrategy, vpunnCostFunc->getStrategyCost(convOp, opStrategy.second.getMCStrategy()));
            if (clusteredOp != nullptr) {
                clusteredOp->removeAttr(VPU::multiClusterStrategy);
            }
        };

        addStrategy(convSOH);
        addStrategy(convSOK);
        addStrategy(convClst);
        addStrategy(convHK);

        storage->setCurrentStrategy(convSOH);
        storage->setBestStrategy(convSOH);
    });

    const auto stateProvider = std::make_shared<DefaultStateProvider>(storage, vpunnCostFunc);
    auto testAlgorithm = createAlgorithm(vpux::VPU::TilingOptions{}, stateProvider, storage);
    testAlgorithm->optimize();

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        EXPECT_EQ(splitOverKStrategy, storage->getBestStrategy(convOp));
    });
}

TEST_F(StateProviderInterfaceTests, StateProvider_InitTemp) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
        %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
        %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
            {ppe = #VPU.PPEStub<>, pad =  #VPU.Padding<bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64>,
            rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
            : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x4xsi32> -> tensor<1x80x28x28xf16, {order = #NHWC}> loc(fused["Conv_100", "t_Convolution"])
        %1 = VPU.Tanh(%0) : tensor<1x80x28x28xf16, {order = #NHWC}>
            -> tensor<1x80x28x28xf16, {order = #NHWC}> loc(fused["Tanh_1", "t_Convolution"])
        %2 = VPU.NCE.MaxPool(%1, %cst)
            {kernel_size = [1, 1], ppe = #VPU.PPEStub<>,
            pad =  #VPU.Padding<bottom = 0, left = 0, right = 0, top = 0>, strides = [1, 1]}
            -> tensor<1x80x28x28xf16, {order = #NHWC}> loc(fused["Maxpool_1", "fused","t_Convolution"])
        return %2 : tensor<1x80x28x28xf16, {order = #NHWC}>
    }
  }
)";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(ArchKind::NPU37XX, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto storage = std::make_shared<OperationStrategies>();

    VPU::Strategy splitOverHStrategy(VPU::MultiClusterStrategy::SplitOverHeight, nullptr);
    VPU::Strategy splitOverKStrategy(VPU::MultiClusterStrategy::SplitOverKernel,
                                     getIntArrayAttr(&ctx, ArrayRef({1, 1, 2, 1})));
    VPU::Strategy multiClusteringStrategy(VPU::MultiClusterStrategy::Clustering, nullptr);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto convSOH = std::make_pair(convOp, splitOverHStrategy);
        auto convSOK = std::make_pair(convOp, splitOverKStrategy);
        auto convClst = std::make_pair(convOp, multiClusteringStrategy);

        storage->addStrategy(convSOH, 1000);
        storage->addStrategy(convSOK, 1200);
        storage->addStrategy(convClst, 1400);
    });

    func->walk([&](VPU::TanhOp tanhOp) {
        auto tanhSOH = std::make_pair(tanhOp, splitOverHStrategy);
        auto tanhSOK = std::make_pair(tanhOp, splitOverKStrategy);
        auto tanhClst = std::make_pair(tanhOp, multiClusteringStrategy);

        storage->addStrategy(tanhSOH, 2000);
        storage->addStrategy(tanhSOK, 2200);
        storage->addStrategy(tanhClst, 2600);
    });

    func->walk([&](VPU::NCEMaxPoolOp maxOp) {
        auto maxSOH = std::make_pair(maxOp, splitOverHStrategy);
        auto maxSOK = std::make_pair(maxOp, splitOverKStrategy);
        auto maxClst = std::make_pair(maxOp, multiClusteringStrategy);

        storage->addStrategy(maxSOH, 3000);
        storage->addStrategy(maxSOK, 3200);
        storage->addStrategy(maxClst, 3400);
    });
    size_t initTemp = getInitialTemperature(storage);
    EXPECT_EQ(1200, initTemp);
}
