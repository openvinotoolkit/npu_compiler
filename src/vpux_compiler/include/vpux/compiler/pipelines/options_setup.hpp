//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/compilation_options.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"
#include "vpux/utils/IE/config.hpp"

#include "intel_npu/config/options.hpp"

namespace vpux {

/// This class serves two purposes:
/// 1. Parsing options from an intel_npu::Config object and overriding specific values depending on platform and
///    compilation mode.
/// 2. From the PipelineStrategy's point of view it is simply a container for a specific type of options.
///
/// OptionsSetup can either parse options from intel_npu::Config or create copies of const VPU::InitCompilerOptions*
/// and const OptionsType*. It owns the objects and derived classes can therefore override specific options without
/// restrictions.
///
/// ConcreteModel has to be the *final* derived class as OptionsSetup will access ConcreteModel::setupOptionsImpl().
/// Make sure that the final derived class exposes its implementation of setupOptionsImpl() to OptionsSetup. This can
/// be done via `friend OptionsSetup<...>`.
///
/// ConcreteModel can choose to only implement one of the setupOptionsImpl() versions. OptionsSetup will then fallback
/// to a default implementation for the not implemented version. However, due to C++'s name resolution you will have to
/// add a `using OptionsSetup<...>::setupOptionsImpl` because as soon as ConcreteModel provides a function named
/// `setupOptionsImpl`, it will hide all parent's functions with the same name [1].
///
/// You can have intermediate derivatives but make sure that the final derivative is always propagated with
/// ConcreteModel!
///
/// [1] https://www.ibm.com/docs/en/zos/2.4.0?topic=scope-name-hiding-c-only
template <class ConcreteModel, class OptionsType>
class OptionsSetup {
public:
    using value_type = OptionsType;

    explicit OptionsSetup(const intel_npu::Config& config): _config(config) {
        _options = parseCompilationModeParams<OptionsType>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(),
                                                           getArchKind(config));
        VPUX_THROW_WHEN(_options == nullptr, "Failed to parse COMPILATION_MODE_PARAMS");

        // it makes sense to call getArchKind/getCompilationMode even though they will be "symbolized" and "stringified"
        // back since parameters will be verified
        auto arch = getArchKind(config);
        auto compilationMode = getCompilationMode(config);
        _initCompilerOptions = std::make_unique<VPU::InitCompilerOptions>(arch, compilationMode, *_options);

        setupOptions();
    }

    // lit-test mode
    explicit OptionsSetup(const VPU::InitCompilerOptions* initCompilerOptions, const OptionsType* options)
            : _options(std::make_unique<OptionsType>()),
              _initCompilerOptions(std::make_unique<VPU::InitCompilerOptions>()) {
        _initCompilerOptions->copyOptionValuesFrom(*initCompilerOptions);
        _options->copyOptionValuesFrom(*options);
        setupOptions();
    }

    virtual ~OptionsSetup() = default;

    const OptionsType& getPipelineOptions() const {
        return *_options;
    }

    const VPU::InitCompilerOptions& getInitCompilerOptions() const {
        return *_initCompilerOptions;
    }

protected:
    // Fallback implementations - make sure they are exposed in ConcreteModel by doing `using Base::setupOptionsImpl;`.
    static void setupLitTestOptionsImpl(OptionsType&, VPU::InitCompilerOptions&) {
        VPUX_THROW("setupOptionsImpl() is not implemented.");
    }

    static void setupOptionsImpl(OptionsType&, VPU::InitCompilerOptions&, const intel_npu::Config&) {
        VPUX_THROW("setupOptionsImpl() is not implemented.");
    }

private:
    void setupOptions() {
        // this way user can setup specific option values
        // for different platforms and compilation modes
        if (_config.has_value()) {
            const auto& configVal = _config.value();

            // Note that all of the following options are explicit OV/Plugin options.
            // Don't parse COMPILATION_MODE_PARAMS again!

            // reuse PSS tests API
            _initCompilerOptions->setAvailableCMXMemory(getAvailableCmx(configVal));

            maybeSetValue(_initCompilerOptions->revisionID, getRevisionID(configVal));
            maybeSetValue(_initCompilerOptions->numberOfDPUGroups, getNumberOfDPUGroups(configVal));
            maybeSetValue(_initCompilerOptions->numberOfDMAPorts, getNumberOfDMAEngines(configVal));
            const auto dynamicQuantization = getCompilerDynamicQuantization(configVal);
            maybeSetValue(_initCompilerOptions->enableWeightsDynamicDequantization, dynamicQuantization);
            if (dynamicQuantization.has_value() && dynamicQuantization.value()) {
                _initCompilerOptions->weightsTableReuseMode = vpux::WeightsTableReuseMode::ENABLED;
            }

            auto optimizationAggressiveEnabled = getQDQOptimizationAggressive(configVal);
            maybeSetValue(_initCompilerOptions->enableQDQOptimizationAggressive, optimizationAggressiveEnabled);

            auto optimizationEnabled = getQDQOptimization(configVal);
            _initCompilerOptions->enableAdaptiveStripping =
                    optimizationAggressiveEnabled.value_or(false) || optimizationEnabled.value_or(false);

            maybeSetValue(_initCompilerOptions->enableProfiling, getPerfCount(configVal));

            const auto& numOfDPUGroups = _initCompilerOptions->numberOfDPUGroups;
            const auto& numOfDMAPorts = _initCompilerOptions->numberOfDMAPorts;

            bool invalidConfig = numOfDPUGroups.hasValue() && numOfDMAPorts.hasValue() &&
                                 numOfDMAPorts.getValue() > numOfDPUGroups.getValue();
            VPUX_THROW_WHEN(invalidConfig,
                            "Requested configuration not supported by runtime. Number of DMA ports ({0}) larger than "
                            "NCE clusters ({1})",
                            numOfDMAPorts.getValue(), numOfDPUGroups.getValue());

            // TODO: #169147 remove this WA
            _options->matchAndCopyOptionValuesFrom(*_initCompilerOptions);

            ConcreteModel::setupOptionsImpl(*_options.get(), *_initCompilerOptions, configVal);
        } else {
            ConcreteModel::setupLitTestOptionsImpl(*_options.get(), *_initCompilerOptions);
        }
    }

    // This class can operate in 2 modes: Either the options stem from an instance of intel_npu::Config. In that case we
    // parse it. In the other case the options come from instances of OptionsType and VPU::InitCompilerOptions. In that
    // case we create a copy.
    std::optional<intel_npu::Config> _config;
    std::unique_ptr<OptionsType> _options;
    std::unique_ptr<VPU::InitCompilerOptions> _initCompilerOptions;

private:
    template <typename OptionType, typename ValType>
    static void maybeSetValue(OptionType& option, std::optional<ValType> value) {
        if (value.has_value()) {
            option = value.value();
        }
    }
};

template <class OptionType, class ValueType>
void overwriteIfUnset(OptionType& option, ValueType&& value) {
    if (!option.hasValue()) {
        Logger::global().info("Overriding option {0} with {1}", option.getArgStr(), value);
        option = value;
    }
}

//
// Below are helper classes for HW-agnostic options settings
// Developer can use them directly or implement new ones for specific platform
//

template <class ConcreteModel, class ArchSpecificOptionsType>
class OptionsSetupBase : public OptionsSetup<ConcreteModel, ArchSpecificOptionsType> {
public:
    using Base = OptionsSetup<ConcreteModel, ArchSpecificOptionsType>;
    using Base::Base;
    // Expose setupOptionsImpl() to OptionsSetup
    friend Base;

protected:
    static void setupLitTestOptionsImpl(ArchSpecificOptionsType& options,
                                        VPU::InitCompilerOptions& initCompilerOptions) {
        setupOptionsCommon(options, initCompilerOptions);
    }

    static void setupOptionsImpl(ArchSpecificOptionsType& options, VPU::InitCompilerOptions& initCompilerOptions,
                                 const intel_npu::Config& config) {
        overwriteIfUnset(options.enableProfiling, config.get<intel_npu::PERF_COUNT>());
        options.updateBatchCompileOptionsFromString(config.get<intel_npu::BATCH_COMPILER_MODE_SETTINGS>());
        setupOptionsCommon(options, initCompilerOptions);
    }

    static void setupOptionsCommon(ArchSpecificOptionsType& options, VPU::InitCompilerOptions& initCompilerOptions) {
        if (initCompilerOptions.enableAdaptiveStripping) {
            options.enableQuantDequantRemoval = true;
            options.enableFuseOutstandingDequant = true;
            options.enableFuseOutstandingQuant = true;
        }
    }
};

template <class ConcreteModel, class ArchSpecificOptionsType>
class WSInitSetupBase : public OptionsSetup<ConcreteModel, ArchSpecificOptionsType> {
public:
    using Base = OptionsSetup<ConcreteModel, ArchSpecificOptionsType>;
    using Base::Base;
    friend Base;

protected:
    static void setupLitTestOptionsImpl(ArchSpecificOptionsType& options, VPU::InitCompilerOptions&) {
        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(ArchSpecificOptionsType& options, VPU::InitCompilerOptions&,
                                 const intel_npu::Config&) {
        setupOptionsCommon(options);
    }

private:
    static void setupOptionsCommon(ArchSpecificOptionsType& options) {
        // E#176454: Profiling is disabled for the @init() function.
        overwriteIfUnset(options.enableProfiling, false);
        // E#176434: remove option
        overwriteIfUnset(options.enableConvertQuantizeOpsToNceOps, false);
        overwriteIfUnset(options.enableAdjustPrecisionPipeline, false);
        overwriteIfUnset(options.enableConvertWeightsToU8I4, false);
        // E#180631: remove option
        overwriteIfUnset(options.forceConvertGatherTo4D, true);
        // E#182190: Enable vertical fusion outlining
        overwriteIfUnset(options.enableVerticalFusionOutlining, false);
    }
};

template <class ConcreteModel, class ArchSpecificOptionsType>
class WSMainSetupBase : public OptionsSetup<ConcreteModel, ArchSpecificOptionsType> {
public:
    using Base = OptionsSetup<ConcreteModel, ArchSpecificOptionsType>;
    using Base::Base;
    friend Base;

protected:
    static void setupLitTestOptionsImpl(ArchSpecificOptionsType& options, VPU::InitCompilerOptions&) {
        setupOptionsCommon(options);
    }

    static void setupOptionsImpl(ArchSpecificOptionsType& options, VPU::InitCompilerOptions&,
                                 const intel_npu::Config& config) {
        setupOptionsCommon(options);
        // Profiling is enabled for the @main() function.
        overwriteIfUnset(options.enableProfiling, config.get<intel_npu::PERF_COUNT>());
    }

private:
    static void setupOptionsCommon(ArchSpecificOptionsType& options) {
        // E#127228 Introduce IE.Swizzle operation
        overwriteIfUnset(options.enableWeightsSwizzling, false);
        // E#127235 Introduce IE.Sparsify operation
        overwriteIfUnset(options.enableWeightsSparsity, false);
        // E#180631: remove option
        overwriteIfUnset(options.forceConvertGatherTo4D, true);
        // E#182190: Enable vertical fusion outlining
        overwriteIfUnset(options.enableVerticalFusionOutlining, false);
    }
};
}  // namespace vpux
