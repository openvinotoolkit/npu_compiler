//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/frontend/ov_batch_detection.hpp"
#include "intel_npu/config/options.hpp"
#include "vpux/compiler/utils/batch.hpp"
#include "vpux/utils/IE/config.hpp"

#include <openvino/core/dimension.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/pass/manager.hpp>

#include <transformations/common_optimizations/dimension_tracking.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

namespace vpux {
bool collectDebatchCoeffDescriptionIfPossible(std::ostream& stream, const ov::PartialShape& shape,
                                              size_t& realBatchShapesCount) {
    size_t i = 0;
    for (i = 0; i < shape.size(); i++) {
        if (shape[i].has_symbol()) {
            stream << DebatchCoeffDescription{Dim{i}, 1}.to_string() << ",";
            if (shape[i] != 1) {
                realBatchShapesCount++;
            }
            break;
        }
    }

    return i < shape.size();
}

bool collectDebatchCoeffDescriptionIfPossible(std::ostream& stream, const ov::Layout& layout,
                                              const ov::PartialShape& shape) {
    if (ov::layout::has_batch(layout)) {
        if (shape[ov::layout::batch_idx(layout)] != 1) {
            stream << DebatchCoeffDescription{Dim{ov::layout::batch_idx(layout)}, 1}.to_string() << ",";
            return true;
        }
    }

    // When we face layout "..." having a rank 0, we don't know what exactly we are dealing with.
    // Let's not deny debatching compile method in this case.
    // It's safe to specify that debatch coeff equal to 1 as if in the case of all coefficients are 1
    // DebatcherPass invocation must not break anything
    if (layout == "..." && shape.rank() == 0) {
        stream << DebatchCoeffDescription{Dim{0}, 1}.to_string() << ",";
        return true;
    }

    return false;
}

bool isExplicitCfgBatchMethodOptionRequested(const intel_npu::Config& config, vpux::Logger& logger) {
    if (!config.has<intel_npu::BATCH_COMPILER_MODE_SETTINGS>()) {
        return false;
    }
    const std::string& compilationModeParams = config.get<intel_npu::BATCH_COMPILER_MODE_SETTINGS>();
    auto batchingAdapterView = BatchCompilerOptionsAdapterView::tryExtractFromString(compilationModeParams);
    if (!batchingAdapterView.has_value()) {
        logger.debug("Cannot parse BatchCompileOptionsAdapter from BATCH_COMPILER_MODE_SETTINGS. Skip explicit batch "
                     "coefficients checking");
        return false;
    }

    if (BatchUnrollOptions::isExplicitlySpecified(batchingAdapterView->get())) {
        logger.debug("Skip batch-network auto detection as \"{0}\" requested",
                     batchingAdapterView->get().batchCompileMethod);
        return true;
    }

    if (DebatcherOptions::isExplicitlySpecified(batchingAdapterView->get())) {
        auto debatcherOptionPtr = DebatcherOptions::create(batchingAdapterView->get());
        VPUX_THROW_WHEN(debatcherOptionPtr == nullptr, "Options must be create-able when it has been set explicitly");
        if (debatcherOptionPtr->debatcherIntputCoeffPartitions !=
            DebatcherOptions::getDefaultDebatchInputCoeffPartitionsValue()) {
            logger.debug("Skip batch-network auto detection as \"{0}\" has been set explicitly: {1}, ",
                         debatcherOptionPtr->debatcherIntputCoeffPartitions.getArgStr(),
                         debatcherOptionPtr->debatcherIntputCoeffPartitions);
            return true;
        }
    }
    return false;
}
std::tuple<bool, std::string> isBatchDetectedByUserLayouts(const std::shared_ptr<ov::Model>& model,
                                                           vpux::Logger& logger) {
    const auto& params = model->get_parameters();
    size_t successfulBatchCoeffCollectionCount = 0;
    std::stringstream debatchingCoefficientsStream;
    for (size_t input_id = 0; input_id < params.size(); input_id++) {
        const auto& input = params[input_id];
        const auto& shape = input->get_partial_shape();
        ov::Layout layout = ov::layout::get_layout(input);
        logger.trace("input: {0}, layout: {1}, shape: {2}", input_id, layout.to_string(), shape.to_string());
        if (collectDebatchCoeffDescriptionIfPossible(debatchingCoefficientsStream, layout, shape)) {
            successfulBatchCoeffCollectionCount++;
        }
    }
    // Although outputs are not used by DebatcherPass while preprocessing,
    // we collect coefficients anyway to enhance debatching algorithm if necessary
    for (const auto& output : model->get_results()) {
        const auto& shape = output->get_output_partial_shape(0);
        ov::Layout layout = ov::layout::get_layout(output);
        logger.trace("output layout: {0}, shape: {1}", layout.to_string(), shape.to_string());
        if (collectDebatchCoeffDescriptionIfPossible(debatchingCoefficientsStream, layout, shape)) {
            successfulBatchCoeffCollectionCount++;
        }
    }

    if (successfulBatchCoeffCollectionCount == 0) {
        return std::make_tuple(false, "Batched network auto detection has been finished: networks input/outputs don't "
                                      "contain information about batch in their layuots");
    }

    // check integrity, do not apply debatching compile method if one or more input/outputs haven't been as batched
    if (successfulBatchCoeffCollectionCount != params.size() + model->get_results().size()) {
        std::stringstream stream;
        stream << "Cannot determine batch dimensions in input/outputs layouts: " << successfulBatchCoeffCollectionCount
               << ". Total inputs: " << params.size() << ", total outputs: " << model->get_results().size()
               << " have been scanned.";
        return std::make_tuple(false, stream.str());
    }
    return std::make_tuple(true, debatchingCoefficientsStream.str());
}

std::tuple<bool, std::string> isBatchDetectedByOVHeuristic(const std::shared_ptr<ov::Model>& model,
                                                           vpux::Logger& logger) {
    std::shared_ptr<ov::Model> testBatchModel = model->clone();

    // Find batch dims and compose debatching coefficients
    try {
        ov::pass::Manager passManager;
        passManager.register_pass<ov::pass::InitNodeInfo>();
        passManager.register_pass<ov::pass::FindBatch>();
        passManager.run_passes(testBatchModel);
    } catch (const std::exception& ex) {
        std::stringstream stream;
        stream << "A batch-network auto detection heuristics have been failed with the error: " << ex.what()
               << "\nPlease specify model inputs/outputs layouts manually so that the compiler could reuse that "
                  "information to bypass "
                  "the auto detection algorithm!"
               << std::endl;
        return std::make_tuple(false, stream.str());
    }

    std::stringstream debatchingCoefficientsStream;
    try {
        size_t successfulBatchCoeffCollectionCount = 0;
        size_t realBatchShapesCount = 0;
        const auto& params = testBatchModel->get_parameters();
        for (size_t input_id = 0; input_id < params.size(); input_id++) {
            const auto& input = params[input_id];
            const auto& shape = input->get_partial_shape();
            logger.trace("input: {0}, has shape: {1}", input_id, shape.to_string());
            if (collectDebatchCoeffDescriptionIfPossible(debatchingCoefficientsStream, shape, realBatchShapesCount)) {
                successfulBatchCoeffCollectionCount++;
            }
        }
        // Although outputs are not used by DebatcherPass while preprocessing,
        // we collect coefficients anyway to enhance debatching algorithm if necessary
        for (const auto& output : testBatchModel->get_results()) {
            const auto& shape = output->get_output_partial_shape(0);
            logger.trace("output has shape: {0}", shape.to_string());
            if (collectDebatchCoeffDescriptionIfPossible(debatchingCoefficientsStream, shape, realBatchShapesCount)) {
                successfulBatchCoeffCollectionCount++;
            }
        }

        if (successfulBatchCoeffCollectionCount == 0) {
            return std::make_tuple(false, "Batched network auto detection has been finished: no any batch found");
        }

        // check integrity, do not apply debatching compile method if one or more input/outputs haven't been as batched
        if (successfulBatchCoeffCollectionCount != params.size() + testBatchModel->get_results().size()) {
            std::stringstream stream;
            stream << "Cannot determine a batch dimension for input/outputs: " << successfulBatchCoeffCollectionCount
                   << ". Total inputs: " << params.size() << ", total outputs: " << testBatchModel->get_results().size()
                   << " scanned.";
            throw std::runtime_error(stream.str());
        }

        // enable debatch only if network has at least single batched shape, with a value of batch dimension != 1
        if (!realBatchShapesCount) {
            debatchingCoefficientsStream << ". Shapes have BATCH dimensions, but debatching is not required";
            return std::make_tuple(false, debatchingCoefficientsStream.str());
        }
    } catch (const std::exception& ex) {
        std::stringstream stream;
        stream << "A batch-network auto detection results can't be correctly interpreted, the error: " << ex.what()
               << " \nPlease specify model inputs/outputs layouts manually so that the compiler could reuse that "
                  "information "
                  "to bypass the auto detection algorithm!"
               << std::endl;
        return std::make_tuple(false, stream.str());
    }
    return std::make_tuple(true, debatchingCoefficientsStream.str());
}

bool isModelSuitableForDebatching(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config,
                                  vpux::Logger& logger) {
    size_t modelOpsNumberEnableThreshold = 10;
    size_t maxBatchNumberDisableLimit = 10;  // TODO E####-59453
    if (config.has<intel_npu::BATCH_COMPILER_MODE_SETTINGS>()) {
        const std::string& compilationModeParams = config.get<intel_npu::BATCH_COMPILER_MODE_SETTINGS>();
        auto batchingAdapterView = BatchCompilerOptionsAdapterView::tryExtractFromString(compilationModeParams);
        if (batchingAdapterView.has_value()) {
            if (auto debatcherOptionPtr = DebatcherOptions::create(batchingAdapterView->get());
                debatcherOptionPtr != nullptr) {
                VPUX_THROW_UNLESS(debatcherOptionPtr, "Being available, DebatcherOptions must have been created");
                if (debatcherOptionPtr->modelOpsNumberEnableThreshold.hasValue()) {
                    modelOpsNumberEnableThreshold = debatcherOptionPtr->modelOpsNumberEnableThreshold;
                }
                if (debatcherOptionPtr->maxBatchNumberDisableLimit.hasValue()) {
                    maxBatchNumberDisableLimit = debatcherOptionPtr->maxBatchNumberDisableLimit;
                }
            }
        }
    }
    logger.debug("Debatch compile method configuration: modelOpsNumberEnableThreshold: {0}, "
                 "maxBatchNumberDisableLimit: {1}. Model input count: {2}, total ops: {3}, output count: {4}",
                 modelOpsNumberEnableThreshold, maxBatchNumberDisableLimit, model->get_parameters().size(),
                 model->get_ops().size(), model->get_results().size());
    size_t modelOpsCount = model->get_ops().size() - model->get_parameters().size() - model->get_results().size();
    if (modelOpsCount < modelOpsNumberEnableThreshold) {
        logger.debug(
                "A model is too small for using \"debatch\" as a batch compilation method. If you are certain in your "
                "intention using \"debatch\" method, please change the threshold explicitly using options: {0}",
                DebatcherOptions::getDefaultOptions());
        return false;
    }

    size_t maxBatchInModel = 0;
    auto params = model->get_parameters();
    for (size_t input_id = 0; input_id < params.size(); input_id++) {
        const auto& input = params[input_id];
        const auto& shape = input->get_partial_shape();
        if (shape.is_dynamic()) {
            continue;
        }
        maxBatchInModel = std::max<size_t>(shape[0].get_length(), maxBatchInModel);
    }
    if (maxBatchInModel > maxBatchNumberDisableLimit) {
        logger.debug(
                "A maximum batch value among inputs of a model: {0} exceeds the limit: {1}. If you are certain in "
                "your intention using \"debatch\" method, please change the threshold explicitly using options: {2}.",
                maxBatchInModel, maxBatchNumberDisableLimit, DebatcherOptions::getDefaultOptions());
        return false;
    }
    return true;
}

std::tuple<intel_npu::Config, bool> autoDetectBatchedModelIfPossible(const std::shared_ptr<ov::Model>& model,
                                                                     const intel_npu::Config& config) {
    std::set<ov::Output<const ov::Node>> batchedInputs;
    std::set<ov::Output<const ov::Node>> batchedOutputs;
    std::set<size_t> sBatchSize;

    vpux::Logger logger("autoDetectBatchedModelIfPossible", getLogLevel(config));

    if (!config.has<intel_npu::BATCH_MODE>()) {
        return {config, false};
    }

    if (config.get<intel_npu::BATCH_MODE>() != ov::intel_npu::BatchMode::COMPILER &&
        config.get<intel_npu::BATCH_MODE>() != ov::intel_npu::BatchMode::AUTO) {
        logger.debug("Config option \"{0}\" is incompatible with batch-network auto detection. Use \"{1}\" or \"{2}\" "
                     "instead",
                     intel_npu::BATCH_MODE::key(), intel_npu::BATCH_MODE::toString(ov::intel_npu::BatchMode::COMPILER),
                     intel_npu::BATCH_MODE::toString(ov::intel_npu::BatchMode::AUTO));
        return {config, false};
    }

    logger.debug("=== Check whether batch compilation requested explicitly ===");
    if (isExplicitCfgBatchMethodOptionRequested(config, logger)) {
        return {config, false};
    }

    bool batchDetected;
    std::string debatchingCoefficients;
    auto enableDebatcher = [&logger, &config](const std::string& debatcherParams) {
        std::string debatchingCompileMethodParams =
                "batch-compile-method=debatch debatcher-settings={debatcher-input-coefficients-partitions=" +
                debatcherParams + "  debatching-inlining-method=naive}";
        logger.debug("Batched network auto detection has been finished. Batch compile method \"debatch\" will be "
                     "employed with params: {0}",
                     debatchingCompileMethodParams);
        auto autoDebatcherOptions =
                BatchCompilerOptionsAdapterView::tryExtractFromString(debatchingCompileMethodParams);
        VPUX_THROW_UNLESS(autoDebatcherOptions.has_value(),
                          "Cannot create BatchCompilerOptionsAdapterView from: \"{0}\"", debatchingCompileMethodParams);

        std::map<std::string, std::string> toUpdate;
        toUpdate[ov::intel_npu::batch_compiler_mode_settings.name()] =
                autoDebatcherOptions->injectInto(config.get<intel_npu::BATCH_COMPILER_MODE_SETTINGS>());

        intel_npu::Config newConfig = config;
        newConfig.update(toUpdate, intel_npu::OptionMode::CompileTime);
        return newConfig;
    };

    logger.debug("=== Run batch-network auto detection heuristics based on model layouts information ===");
    std::tie(batchDetected, debatchingCoefficients) = isBatchDetectedByUserLayouts(model, logger);
    if (batchDetected) {
        if (isModelSuitableForDebatching(model, config, logger)) {
            return {enableDebatcher(debatchingCoefficients), true};
        }
        return {config, false};
    }
    logger.debug("{0}", debatchingCoefficients);

    logger.debug("=== Run OpenVINO batch-network auto detection heuristics ===");
    std::tie(batchDetected, debatchingCoefficients) = isBatchDetectedByOVHeuristic(model, logger);
    if (batchDetected) {
        if (isModelSuitableForDebatching(model, config, logger)) {
            return {enableDebatcher(debatchingCoefficients), true};
        }
        return {config, false};
    }

    return {config, false};
}

bool checkCfgOnBatchOptionConsistency(const intel_npu::Config& config, std::ostream& outDescr) {
    if ((config.get<intel_npu::BATCH_MODE>() == ov::intel_npu::BatchMode::PLUGIN) &&
        config.has<intel_npu::BATCH_COMPILER_MODE_SETTINGS>()) {
        outDescr << "BATCH_COMPILER_MODE_SETTINGS can't be specified while BATCH_MODE is PLUGIN";
        return false;
    }
    return true;
}

std::shared_ptr<ov::Model> debatchDynamicModel(const std::shared_ptr<ov::Model>& origModel, Logger& logger) {
    bool batchDetected = false;
    std::string debatchingCoefficients;
    auto debatchModel = [&logger, &origModel](const std::string& debatcherParams) -> std::shared_ptr<ov::Model> {
        logger.debug("Trying to apply model debatching coefficients: {0}", debatcherParams);
        auto debatchedModel = origModel->clone();
        auto coefficients = DebatchCoefficients::create(debatcherParams);
        VPUX_THROW_UNLESS(coefficients.has_value(), "Cannot create DebatchCoefficients from string: {0}",
                          debatcherParams);
        size_t inputIdx = 0;
        std::map<std::string, ov::PartialShape> newShapes;
        for (auto&& item : debatchedModel->get_parameters()) {
            auto layout = item->get_layout();
            auto partShape = item->get_partial_shape();
            if (!ov::layout::has_batch(layout)) {
                std::optional<DebatchCoeffDescription> coeff = coefficients->getCoefficient(inputIdx);
                if (coeff.has_value()) {
                    partShape[coeff.value().batchPositionIndex.ind()] = 1;
                }
            } else {
                partShape[ov::layout::batch_idx(layout)] = 1;
            }
            logger.debug("Input: {0} will use the new shape: {1}", item->get_friendly_name(), partShape.to_string());
            newShapes.emplace(item->get_friendly_name(), partShape);
            inputIdx++;
        }
        debatchedModel->reshape(newShapes);
        return debatchedModel;
    };

    logger.debug("=== Run batch-network auto detection heuristics based on model layouts information ===");
    std::tie(batchDetected, debatchingCoefficients) = isBatchDetectedByUserLayouts(origModel, logger);
    if (batchDetected) {
        return debatchModel(debatchingCoefficients);
    }

    logger.debug("=== Run OpenVINO batch-network auto detection heuristics ===");
    std::tie(batchDetected, debatchingCoefficients) = isBatchDetectedByOVHeuristic(origModel, logger);
    if (batchDetected) {
        return debatchModel(debatchingCoefficients);
    }

    return {nullptr};
}
}  // namespace vpux
