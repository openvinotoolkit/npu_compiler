// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include <gflags/gflags.h>

#include <openvino/opsets/opset3.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include "openvino/core/version.hpp"
#include "openvino/openvino.hpp"

#include <string>
#include <vector>

DEFINE_string(output, "", "Name for output IR (<name>.xml + <name>.bin)");
DEFINE_string(inputs_size, "", "Size of each input. Example: \"100 200\"");
DEFINE_string(outputs_size, "", "Size of each input. Example: \"100 200\"");

void parseCommandLine(int argc, char* argv[]) {
    std::ostringstream usage;
    usage << "Usage: " << argv[0] << " [<options>]";
    gflags::SetUsageMessage(usage.str());

    std::ostringstream version;
    version << ov::get_openvino_version();
    gflags::SetVersionString(version.str());

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_output.empty() || FLAGS_inputs_size.empty() || FLAGS_outputs_size.empty()) {
        OPENVINO_THROW("Not enough parameters. Please check help (--help).");
    }

    std::cout << "Parameters:" << std::endl;
    std::cout << "    Output file:     " << FLAGS_output + ".xml" << std::endl;
    std::cout << "    Inputs size:     " << FLAGS_inputs_size << std::endl;
    std::cout << "    Outputs size:    " << FLAGS_outputs_size << std::endl;
    std::cout << std::endl;
}

std::vector<size_t> parseShapes(const std::string& shapesStr) {
    std::vector<size_t> shapes;
    std::stringstream shapesSS(shapesStr);
    size_t shape{};
    while (shapesSS >> shape) {
        shapes.push_back(shape);
    }
    return shapes;
}

int main(int argc, char* argv[]) {
    try {
        parseCommandLine(argc, argv);

        // fp16 elemType by default
        const auto elementType = ov::element::f16;

        const std::vector<size_t> inputShapes = parseShapes(FLAGS_inputs_size);
        const std::vector<size_t> outputShapes = parseShapes(FLAGS_outputs_size);

        if (inputShapes.size() != outputShapes.size()) {
            OPENVINO_THROW("Number of inputs must be equal to the number of outputs.");
        }

        // Only C layout as input supported
        const size_t dims = 1;

        std::vector<std::shared_ptr<ov::opset3::Parameter>> parameters;
        std::vector<std::shared_ptr<ov::opset3::StridedSlice>> stridedSlices;
        std::vector<std::shared_ptr<ov::opset3::Result>> results;

        const std::vector<int64_t> beginMask = {0, 0, 0, 0};
        const std::vector<int64_t> endMask = {0, 0, 0, 0};

        for (size_t i = 0; i < outputShapes.size(); ++i) {
            if (outputShapes[i] > inputShapes[i]) {
                OPENVINO_THROW("Output size for each pair must be smaller or equal to input.");
            }

            auto data = std::make_shared<ov::opset3::Parameter>(elementType, ov::Shape({inputShapes[i]}));
            const std::string friendlyInputName = "Input" + std::to_string(i);
            data->set_friendly_name(friendlyInputName);
            data->output(0).get_tensor().set_names({friendlyInputName});

            const auto begin = ov::opset3::Constant::create(ov::element::i64, ov::Shape{dims}, {0});
            const auto end = ov::opset3::Constant::create(ov::element::i64, ov::Shape{dims}, {outputShapes[i]});
            const auto stride = ov::opset3::Constant::create(ov::element::i64, ov::Shape{dims}, {1});

            auto ss = std::make_shared<ov::opset3::StridedSlice>(data, begin, end, stride, beginMask, endMask);

            auto res = std::make_shared<ov::opset3::Result>(ss);

            auto friendlyResultName = std::string("Result") + std::to_string(i);
            res->set_friendly_name(friendlyResultName);
            res->output(0).get_tensor().set_names({std::move(friendlyResultName)});

            parameters.push_back(std::move(data));
            stridedSlices.push_back(std::move(ss));
            results.push_back(std::move(res));
        }

        auto network = std::make_shared<ov::Model>(ov::ResultVector(std::move(results)),
                                                   ov::ParameterVector(std::move(parameters)));

        ov::pass::Manager passManager;

        const std::string graphName = FLAGS_output;
        passManager.register_pass<ov::pass::Serialize>(graphName + ".xml", graphName + ".bin");
        passManager.run_passes(std::move(network));
    }  // try
    catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
