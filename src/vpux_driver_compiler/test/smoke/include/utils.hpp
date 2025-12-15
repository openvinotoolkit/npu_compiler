//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "openvino/core/layout.hpp"
#include "openvino/core/model.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/core.hpp"

bool hasOption(const char* option, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == option) {
            std::cout << "For compatibility with the old version, check the parameter in command:"
                      << std::string(argv[i]) << std::endl;
            return true;
        }
    }
    return false;
}

std::string getPlatform(const std::string& device) {
    if (device == "NPU.3720") {
        return "3720";
    } else if (device == "NPU.4000" || device == "NPU") {
        return "4000";
    } else if (device == "NPU.5010") {
        return "5010";
    } else {
        throw std::runtime_error("Target device is unknown.");
    }
}

void boundDynamicShape(std::shared_ptr<ov::Model>& model) {
    for (auto&& item : model->get_parameters()) {
        auto shape = item->get_partial_shape();
        if (shape.is_static()) {
            continue;
        }
        auto rank = shape.rank();
        if (rank.is_dynamic()) {
            throw std::logic_error("Rank \"" + rank.to_string() + "\" of the shape \"" + shape.to_string() +
                                   "\" is dynamic, which is not supported by NPU.");
        }
        auto layout = item->get_layout();
        if (!ov::layout::has_batch(layout)) {
            item->set_layout(ov::Layout(layout.to_string().insert(1, "N,")));
            layout = item->get_layout();
        }
        if (shape[ov::layout::batch_idx(layout)].is_dynamic()) {
            std::cout << "WARNING: Shape \"" + shape.to_string() + "\"" +
                                 " has dynamic batch size, which is not supported by NPU.\n"
                                 "         Setting batch to 1 forcefully."
                      << std::endl;
            ov::set_batch(model, 1);
        }
        shape = item->get_partial_shape();
        if (shape.is_dynamic()) {
            throw std::logic_error("Model's input shape \"" + shape.to_string() + "\"" +
                                   " is dynamic, which is not supported by NPU.");
        }
    }
}

struct InputInfo {
    ov::element::Type type;
    ov::PartialShape partialShape;
    ov::Shape dataShape;
    ov::Layout layout;
};

using InputsInfo = std::map<std::string, InputInfo>;

std::string parameterNameToTensorName(const std::string& name, const std::vector<ov::Output<ov::Node>>& inputsInfo) {
    auto countName = std::any_of(inputsInfo.begin(), inputsInfo.end(), [name](const ov::Output<ov::Node>& port) {
        return port.get_names().count(name) > 0;
    });
    if (countName) {
        return name;
    } else {
        auto inputInfo = std::find_if(inputsInfo.begin(), inputsInfo.end(), [name](const ov::Output<ov::Node>& port) {
            return name == port.get_node()->get_friendly_name();
        });
        if (inputInfo == inputsInfo.end()) {
            throw std::runtime_error("Provided I/O name \"" + name +
                                     "\" is not found neither in tensor names nor in nodes names.");
        }
        return inputInfo->get_any_name();
    }
}

std::map<std::string, std::vector<std::string>> parseInputParameters(
        const std::string& parameterString, const std::vector<ov::Output<ov::Node>>& input_info) {
    // Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all
    // inputs)
    std::map<std::string, std::vector<std::string>> returnValue;
    std::string searchString = parameterString;
    auto start_pos = searchString.find_first_of('[');
    auto input_name = searchString.substr(0, start_pos);
    while (start_pos != std::string::npos) {
        auto end_pos = searchString.find_first_of(']');
        if (end_pos == std::string::npos) {
            break;
        }
        input_name = searchString.substr(0, start_pos);
        auto input_value = searchString.substr(start_pos + 1, end_pos - start_pos - 1);
        if (!input_name.empty()) {
            returnValue[parameterNameToTensorName(input_name, input_info)].push_back(std::move(input_value));
        } else {
            for (auto& item : input_info) {
                returnValue[item.get_any_name()].push_back(input_value);
            }
        }
        searchString = searchString.substr(end_pos + 1);
        if (searchString.empty() || (searchString.front() != ',' && searchString.front() != '[')) {
            break;
        }
        if (searchString.front() == ',') {
            if (searchString.length() > 1) {
                searchString = searchString.substr(1);
            } else {
                throw std::logic_error("Can't parse input parameter string, there is nothing after the comma " +
                                       parameterString);
            }
        }
        start_pos = searchString.find_first_of('[');
    }
    if (!searchString.empty()) {
        throw std::logic_error("Can't parse input parameter string: " + parameterString);
    }
    return returnValue;
}

void reshape(const ov::OutputVector& inputsInfo, InputsInfo& infoMap, std::shared_ptr<ov::Model>& model,
             const std::string& shapeString, int overrideModelBatchSize) {
    std::vector<InputsInfo> infoMaps;
    if (!shapeString.empty()) {
        std::map<std::string, std::vector<std::string>> shapesMap = parseInputParameters(shapeString, inputsInfo);

        if (overrideModelBatchSize != 1) {
            throw std::logic_error(R"(Incompatible params: "shape" and "override_model_batch_size")");
        }
        for (auto& item : inputsInfo) {
            InputInfo info;
            auto name = item.get_any_name();

            if (!shapesMap.empty()) {
                if (shapesMap.count(name)) {
                    if (shapesMap.at(name).size() > 1) {
                        // Example: -shape input1[..][..]
                        throw std::logic_error("shape command line parameter doesn't support multiple "
                                               "shapes for one input.");
                    }
                    info.partialShape = shapesMap.at(name)[0];
                } else {
                    info.partialShape = item.get_partial_shape();
                }
            }
            infoMap[name] = std::move(info);
            infoMaps.push_back(infoMap);
        }
        std::map<std::string, ov::PartialShape> newShapes;
        for (auto& item : infoMaps) {
            for (auto& map : item) {
                if (!newShapes.count(map.first)) {
                    newShapes[map.first] = map.second.partialShape;
                }
            }
        }
        model->reshape(newShapes);
    } else {
        boundDynamicShape(model);
    }
}

void setModelBatch(std::shared_ptr<ov::Model>& model, uint32_t batch = 1) {
    if (batch == 1) {
        return;
    }
    std::cout << "Configuring model batch: " << batch << std::endl;
    for (auto&& item : model->get_parameters()) {
        auto shape = item->get_partial_shape();
        auto rank = shape.rank();
        if (rank.is_dynamic()) {
            throw std::logic_error("Rank \"" + rank.to_string() + "\" of the shape \"" + shape.to_string() +
                                   "\" is dynamic, which is not supported by NPU.");
        }
        auto layout = item->get_layout();
        if (!ov::layout::has_batch(layout)) {
            item->set_layout(ov::Layout(layout.to_string().insert(1, "N,")));
            layout = item->get_layout();
        }
        if (shape[ov::layout::batch_idx(layout)].is_dynamic()) {
            throw std::logic_error("ERROR: Shape \"" + shape.to_string() + "\"" +
                                   " has dynamic batch size, which is not supported by NPU.\n"
                                   "Cannot apply fixed batch: " +
                                   std::to_string(batch) +
                                   ". Please remove the parameter from config: \"override_model_batch_size\"");
        }
        ov::set_batch(model, batch);
    }
}

void printInputAndOutputsInfoShort(const ov::Model& network) {
    std::cout << "Network inputs:" << std::endl;
    for (auto&& param : network.get_parameters()) {
        auto l = param->get_layout();
        std::cout << "    " << param->get_friendly_name() << " : " << param->get_element_type() << " / "
                  << param->get_layout().to_string() << " / " << param->get_partial_shape().to_string() << std::endl;
    }
    std::cout << "Network outputs:" << std::endl;
    for (auto&& result : network.get_results()) {
        std::cout << "    " << result->get_friendly_name() << " : " << result->get_element_type() << " / "
                  << result->get_layout().to_string() << " / " << result->get_output_partial_shape(0).to_string()
                  << std::endl;
    }
}

int getName(const std::string& inputblobFileName, std::string& outputBlobFileName) {
    if (inputblobFileName.empty()) {
        std::cout << "The input model file is empty!" << std::endl;
        return -2;
    }
    std::string blobFileName(inputblobFileName);

    const std::string allocatedBlobFileNameSuffix = ".allocator";
    outputBlobFileName = blobFileName + allocatedBlobFileNameSuffix;

    if (outputBlobFileName.empty()) {
        std::cout << "failed to allocate memory for allocated blob file names" << std::endl;
        return -3;
    }

    return 0;
}
