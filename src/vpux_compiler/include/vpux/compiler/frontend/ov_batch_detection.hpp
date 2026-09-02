//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/layout.hpp>
#include <openvino/core/model.hpp>
#include <openvino/core/partial_shape.hpp>

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/utils/logger/logger.hpp"
#include "vpux/utils/ov/options.hpp"

namespace vpux {

/**
 * @brief finds a batch dimension in @layout and @shape. In case if the batch is
 * detected and is not a unit array (N != 1), the function returns true
 *
 * @param layout - the layout of models input/output
 * @param shape - the shape of the appropriate input/output
 */
bool hasExplicitBatchInLayout(const ov::Layout& layout, const ov::PartialShape& shape);

/**
 * @brief checks if layout is set explicitly and doesn't contain a batch dimension
 * Return true, if no batch dimension exists
 *
 * @param layout - the layout of models input/output
 * @param shape - the shape of the appropriate input/output
 */
bool hasExplicitBatchInLayoutForbidden(const ov::Layout& layout);

/**
 * @brief analyze @shape of unknown layout whether it contains a batch dimension. In case if the batch is being
 * detected, the function increments @realBatchShapesCount when the batch dimension is not a unit array (N != 1). The
 * coefficient descriptions allow us to build DebatchCoefficients in order to specify DebatcherPass details precisely
 *
 * @param stream - an output stream used to collect detection info
 * @param shape - an input shape for the analysis
 * @param realBatchShapesCount - an output counter of detected N != 1 batch dimensions.
 *
 * @details Both the @realBatchShapesCount counter and the fucntion result will be involed as a decision making
 * routine arguments in order to decide whether a model is batched model or not
 */
bool collectDebatchCoeffDescriptionIfPossible(std::ostream& stream, const ov::PartialShape& shape,
                                              size_t& realBatchShapesCount);
/**
 * @brief analyze @shape based on its @layout whether it contains a batch dimension. In case if the batch is being
 * detected, the function increments @realBatchShapesCount when the batch dimension is not a unit array (N != 1) The
 * coefficient descriptions allow us to build DebatchCoefficients in order to specify DebatcherPass details precisely
 *
 * @param stream - an output stream used to collect detection info
 * @param layout - a shape layout
 * @param shape - an input shape for the analysis
 *
 * @details counter of N != 1 batch dimensions is not using here, as both @layout property existence and a function
 * result are enough to make a decision whether a model is batched or not.
 */
bool collectDebatchCoeffDescriptionIfPossible(std::ostream& stream, const ov::Layout& layout,
                                              const ov::PartialShape& shape);

/**
 * @brief analyze @config on explicit batch compile options provided externally.
 * In case of either batch-compile-method=unroll or batch-compile-method=debatch (with extended params) detected, the
 * method returns true, which bypasses further audo-batch recognition algorithm. If no other batch option presence were
 * detected, the function returns false
 *
 * @param config - a configuration instance being analized on explicit batch compilation options request
 * @param logger - an instance of a logger
 *
 * @details For further details please check out
 * `tests/unit/vpux_compiler/frontend/auto_batch_compiler_detection_utils.cpp` where you can find information about
 * suitable combinations of params and options
 */
bool isExplicitCfgBatchMethodOptionRequested(const vpux::OV::Config& config, vpux::Logger& logger);

/* @brief checks whether inputs and outputs layouts of a @model have no batch dimensions.
 * If there are no batch dimensions neither in inputs not outputs, which have their layouts set explicitly,
 * the function deems that the batch processing is forbidden and returns true.
 * If at least one N-dimension is explicitly specified, then it assumes that batch processing is enabled,
 * so that auto-detection of batch will go on the deep inspection
 *
 * @param model - a model for batch detection
 * @param logger - an instance of a logger
 *
 * @details For further details please check out
 * `tests/unit/vpux_compiler/frontend/auto_batch_compiler_detection_utils.cpp` where you can find information about
 * suitable combinations of params and options
 */
bool isBatchForbiddenExplicitly(const std::shared_ptr<ov::Model>& model, vpux::Logger& logger);

/**
 * @brief the function determines whether a model has a batch dimension based on complemented layout information.
 * If layouts are not specified or inconsistent that it returns a pair of false and fail-description.
 * If layout are specified and batch dimensions detected, then a pair of true and recommended debatch-coefficients is
 * being returned.
 *
 * @param model - a model for batch detection
 * @param logger - an instance of a logger
 *
 * @details For further details please check out `tests/unit/vpux_compiler/frontend/auto_batch_compiler_detection.cpp`
 * where you can find information about suitable combinations of params and options in @config
 */
std::tuple<bool, std::string> isBatchDetectedByUserLayouts(const std::shared_ptr<ov::Model>& model,
                                                           vpux::Logger& logger);

/**
 * @brief the function determines whether a model has a batch dimension based on OpenVINO intrinsic heuristics.
 * If batch dimensions detected, then a pair of true and recommended debatch-coefficients is being returned.
 * Otherwise a pair of false and an error description
 *
 * @param model - a model for batch detection
 * @param logger - an instance of a logger
 *
 * @details For further details please check out `tests/unit/vpux_compiler/frontend/auto_batch_compiler_detection.cpp`
 * where you can find information about suitable usecases
 */
std::tuple<bool, std::string> isBatchDetectedByOVHeuristic(const std::shared_ptr<ov::Model>& model,
                                                           vpux::Logger& logger);

/**
 * @brief the function introduces additional criteria determining whether a model is suitable for debatching or not.
 * If batch dimensions have been detected and they have an enormous value (2000 for example), then it usually means that
 * we are trying to compile test models rather than a regular model. Although no hypothetical limitations are declared,
 * it also means that dealing with a batch of 2000 size will cause massive outlining-inlining of functions, which
 * typically disrupts common validation flow by inflating of total execution time drastically. So that to overcome the
 * problem this function was introduced. It checks a model layer count and a max batch size of each model input. If a
 * model is "too thin" (the layer count is lesser than the expected threshold) or a max batch number is enormous
 * (exceeds the limit) then we assume that the model is not suitable for debatching. The threshold and the limit may be
 * turned off explicitly by using a non default BATCH_COMPILER_MODE_SETTING. Please look at `DebatcherOptions` in order
 * to bypass these checks.
 *
 * @param model - a model for batch detection
 * @param config - a config with limits specified
 * @param logger - an instance of a logger
 *
 * @details For further details please check out `tests/unit/vpux_compiler/frontend/auto_batch_compiler_detection.cpp`
 * where you can find information about suitable usecases
 */
bool isModelSuitableForDebatching(const std::shared_ptr<ov::Model>& model, const vpux::OV::Config& config,
                                  vpux::Logger& logger);

/**
 * @brief the function tries to recognize a model as a batched model, based on criterias embodied by the functions
 * above. If success, then it complements the @config by putting additional information, which is supposed to be
 * consumed by a compiler pipeline, in order to turn on batch compilation in most efficient way. Such pre-analysis is
 * important, because intrincis compiler components like passes have no information regarding proper tensor partitioning
 * (like data layout), which is only exist in ov::Model. Having batch detected successfully, returns a new config, which
 * is a copy of the initial config but updated with the refined option NPU_BATCH_COMPILER_MODE_SETTINGS. Otherwise it
 * returns a pair of the old config and false flag
 *
 * @param model - a model for batch detection
 * @param config - an instance of config representing an initial set of options used for a model compiation
 *
 * @details For further details please check out tests/unit/vpux_compiler/frontend/auto_batch_compiler_detection.cpp
 * where you can find information about suitable combinations of @model and @config
 */
std::tuple<vpux::OV::Config, bool> autoDetectBatchedModelIfPossible(const std::shared_ptr<ov::Model>& model,
                                                                    const vpux::OV::Config& config);

/**
 * @brief the function analyses existing batch @config options requested explicitly and checks its controversy.
 * In case of failure the function returns a pair of false and an error descrption
 *
 * @param model - a model for batch detection
 * @param config - an instance of config representing a set of options used for a model compiation
 *
 * @details For further details please check out
 * tests/unit/vpux_compiler/frontend/auto_batch_compiler_detection.cpp
 * where you can find information about suitable combination of configuration options
 */
bool checkCfgOnBatchOptionConsistency(const vpux::OV::Config& config, std::ostream& outDescr);

std::shared_ptr<ov::Model> debatchDynamicModel(const std::shared_ptr<ov::Model>& origModel, Logger& logger);
}  // namespace vpux
