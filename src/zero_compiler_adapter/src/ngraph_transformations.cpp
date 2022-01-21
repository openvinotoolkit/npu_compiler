//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "ngraph_transformations.h"
#include <file_reader.h>
#include <chrono>
#include <ngraph/pass/manager.hpp>
#include <transformations/serialize.hpp>

#include "vpux/passes/convert_MVN6_to_MVN1.hpp"

namespace vpux {
namespace zeroCompilerAdapter {
namespace ngraphTransformations {

void applyLoweringPasses(const std::shared_ptr<ngraph::Function>& netGraph, size_t opsetVersion) {
    //    _logger->info("Opset version: {}", opsetVersion.version);
    if (opsetVersion < 6) {
        lowerFromOpset6(netGraph);
        //        _logger->debug("lowerFromOpset6 called");
    }
}

bool isFunctionSupported(const std::shared_ptr<const ngraph::Function>& netGraph, size_t opsetVersion) {
    size_t highestVersion = 0;
    for (const auto& op : netGraph->get_ops()) {
        if (op->get_version() > highestVersion) {
            highestVersion = op->get_version();
        }
    }

    if (highestVersion > opsetVersion) {
        return false;
    }
    return true;
}

IR serializeToIR(const std::shared_ptr<ngraph::Function>& netGraph) {
    //    const std::unique_ptr<vpu::Logger> _logger = std::unique_ptr<vpu::Logger>(new vpu::Logger(
    //            "ngraphTransformations", vpu::LogLevel::Debug /*_config.logLevel()*/, vpu::consoleOutput()));
    const auto passConfig = std::make_shared<ngraph::pass::PassConfig>();
    ngraph::pass::Manager manager(passConfig);

    std::stringstream xmlStream, weightsStream;
    manager.register_pass<ngraph::pass::Serialize>(xmlStream, weightsStream);
    manager.run_passes(netGraph);

    const size_t xmlSize = vpu::KmbPlugin::utils::getFileSize(xmlStream);
    const size_t weightsSize = vpu::KmbPlugin::utils::getFileSize(weightsStream);
    std::vector<char> xmlBlob(xmlSize), weightsBlob(weightsSize);

    xmlStream.read(xmlBlob.data(), xmlSize);
    weightsStream.read(weightsBlob.data(), weightsSize);

    //    _logger->info("|| Timer ||;ngraphTransformations::serializeToIR (ms);\t{}",
    //                  std::chrono::duration_cast<ms>(finish - start).count());
    return {xmlBlob, weightsBlob};
}

void lowerFromOpset6(const std::shared_ptr<ngraph::Function>& netGraph) {
    const auto passConfig = std::make_shared<ngraph::pass::PassConfig>();
    ngraph::pass::Manager manager(passConfig);

    manager.register_pass<vpux::passes::ConvertMVN6toMVN1>();

    manager.run_passes(netGraph);
}

}  // namespace ngraphTransformations
}  // namespace zeroCompilerAdapter
}  // namespace vpux
