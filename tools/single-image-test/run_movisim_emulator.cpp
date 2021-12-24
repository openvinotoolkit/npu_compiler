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

#ifdef ENABLE_MOVISIM

#include "run_movisim_emulator.hpp"

#include "utils.hpp"

#include <fstream>
#include <iostream>

namespace ms {

template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    auto buf = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

InferenceEngine::BlobMap runMoviSimEmulator(InferenceEngine::ExecutableNetwork& exeNet, std::string pathToNetworkBlob,
                                            const std::vector<std::string>& dumpedInputsPaths,
                                            const std::map<std::string, size_t>& outputIndexes) {
    std::string runMovisimScriptLocation =
            std::getenv("MOVISIM_SCRIPT_LOCATION") != nullptr ? std::getenv("MOVISIM_SCRIPT_LOCATION") : "";
    if (runMovisimScriptLocation.empty()) {
        throw std::logic_error(
                "MOVISIM_SCRIPT_LOCATION evnironment variable hasn't been provided. Please check your environment");
    }
    std::string Python_EXECUTABLE = std::getenv("PYTHONPATH") != nullptr ? std::getenv("PYTHONPATH") : "python3";
    if (Python_EXECUTABLE.empty()) {
        throw std::logic_error("PYTHONPATH location can't be defined. Please check your environment");
    }
    std::string runScriptCommand =
            Python_EXECUTABLE + " " + runMovisimScriptLocation + "/run_MoviSim.py" + " -n" + pathToNetworkBlob;
    // provide inputs
    for (const auto& i : dumpedInputsPaths) {
        runScriptCommand += " -i" + i;
    }

    //  provide outputs
    std::vector<std::string> outputFiles;
    std::string outputFilePattern = dumpedInputsPaths[0];

    auto index = outputFilePattern.find("input_0");
    if (index != std::string::npos) {
        outputFilePattern.replace(index, 7, "movisim_output_%d");
    }

    for (size_t i = 0; i < exeNet.GetOutputsInfo().size(); ++i) {
        outputFiles.push_back(string_format(outputFilePattern, i));
        runScriptCommand += " -o" + outputFiles.back();
    }

    // run python script
    std::cout << "try to run:\n" << runScriptCommand << std::endl;
    FILE* pipe;
    char scriptOutputChunk[1024];
    pipe = popen(runScriptCommand.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    // read output from script and print it
    while (fgets(scriptOutputChunk, sizeof(scriptOutputChunk), pipe) != nullptr) {
        printf("%s", scriptOutputChunk);
    }

    pclose(pipe);

    // load movisim outputs into network output
    InferenceEngine::BlobMap output_blobs;
    for (const auto& out : exeNet.GetOutputsInfo()) {
        auto outputIndex = outputIndexes.at(out.first);
        std::cout << "load movisim_output: " << outputFiles[outputIndex] << "\n\tinto: " << out.first
                  << " with precision: " << out.second->getTensorDesc().getPrecision() << std::endl;
        auto blob = loadBlob(out.second->getTensorDesc(), outputFiles[outputIndex]);
        output_blobs.insert({out.first, blob});
    }

    return output_blobs;
}

}  // namespace ms

#endif  // ENABLE_MOVISIM
