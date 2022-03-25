#include "include/mcm/target/target_descriptor.hpp"

std::string mv::TargetDescriptor::toString(Target target)
{
    switch (target)
    {

        case Target::ma2490:
            return "ma2490";

        case Target::ma3100:
            return "ma3100";

        case Target::ma3720:
            return "ma3720";

        default:
            return "unknown";

    }
}

mv::Target mv::TargetDescriptor::toTarget(const std::string& str)
{
    if (str == "ma2490")
        return Target::ma2490;

    if (str == "ma3100")
        return Target::ma3100;

    if (str == "ma3720")
        return Target::ma3720;

    return Target::Unknown;
}

mv::TargetDescriptor::TargetDescriptor(const std::string& filePath) :
target_(Target::Unknown),
globalDType_("Float16"),
codecDef_(),
generalConfigs_({false, false, 7, false, PWLTableMap()})//KMB default: HW PRELU MULT is I8, so 7 precision bits are available
{

    if (!filePath.empty())
        if (!load(filePath))
            throw ArgumentError(*this, "filePath", filePath,
                "Unable to parse target descriptor - error reading or invalid");

}

void mv::TargetDescriptor::reset()
{
    target_ = Target::Unknown;
    globalDType_ = DType("Float16");
    memoryDefs_.clear();
    nceDefs_.clear();
    dtypeSupport_.clear();
    processorDefs_.clear();
    workloadConfigs_.clear();
    generalConfigs_.floatScaleTable = false;
    generalConfigs_.allowMultipleInputScales = false;
    generalConfigs_.leakyAccuracyBits = 7;
    generalConfigs_.pwlEnabled = true;
    generalConfigs_.pwlTables.clear();
}

bool mv::TargetDescriptor::load(const std::string& filePath)
{

    JSONTextParser parser(jsonParserBufferLenght_);
    json::Object jsonDescriptor;

    try
    {

        json::Value jsonRoot;
        if (!parser.parseFile(filePath, jsonRoot))
        {
            throw ArgumentError(*this, "filePath", filePath,
                "Unable to parse target descriptor - error reading");
        }
        if (jsonRoot.valueType() != json::JSONType::Object)
            return false;
        else
            jsonDescriptor = jsonRoot.get<json::Object>();

    }
    catch (ParsingError& e)
    {
        return false;
    }

    if (!jsonDescriptor.hasKey("target"))
        return false;
    else
    {
        target_ = toTarget(jsonDescriptor["target"].get<std::string>());
        if (target_ == Target::Unknown)
            return false;
    }

    if (!jsonDescriptor.hasKey("dtype"))
    {
        reset();
        return false;
    }

    if (jsonDescriptor.hasKey("general"))
    {
        if (jsonDescriptor["general"].valueType() != json::JSONType::Object)
        {
            reset();
            return false;
        }
        else
        {
            if (jsonDescriptor["general"].hasKey("floatScaleTable"))
            {
                if (jsonDescriptor["general"]["floatScaleTable"].valueType() != json::JSONType::Bool)
                {
                    reset();
                    return false;
                }
                else
                {
                    generalConfigs_.floatScaleTable = jsonDescriptor["general"]["floatScaleTable"].get<bool>();
                }
            }
            if (jsonDescriptor["general"].hasKey("pwlEnabled"))
            {
                if (jsonDescriptor["general"]["pwlEnabled"].valueType() != json::JSONType::Bool)
                {
                    reset();
                    return false;
                }
                else
                {
                    generalConfigs_.pwlEnabled = jsonDescriptor["general"]["pwlEnabled"].get<bool>();
                }
            }

            if (jsonDescriptor["general"].hasKey("allowMultipleInputScales"))
            {
                if (jsonDescriptor["general"]["allowMultipleInputScales"].valueType() != json::JSONType::Bool)
                {
                    reset();
                    return false;
                }
                else
                {
                    generalConfigs_.allowMultipleInputScales = jsonDescriptor["general"]["allowMultipleInputScales"].get<bool>();
                }
            }
            if (jsonDescriptor["general"].hasKey("leakyAccuracyBits"))
            {
                if (jsonDescriptor["general"]["leakyAccuracyBits"].valueType() != json::JSONType::NumberInteger)
                {
                    reset();
                    return false;
                }
                else
                {
                    generalConfigs_.leakyAccuracyBits = jsonDescriptor["general"]["leakyAccuracyBits"].get<long long>();
                }
            }
			if (jsonDescriptor["general"].hasKey("customPWLTable")) {
                if (jsonDescriptor["general"]["customPWLTable"].valueType() != json::JSONType::Array) {
                    reset();
                    return false;
                }

                auto pwl_tables = jsonDescriptor["general"]["customPWLTable"];
                for (size_t i = 0; i < pwl_tables.size(); ++i) {
                    if (!pwl_tables[i].hasKey("activation") ||
                        pwl_tables[i]["activation"].valueType() != json::JSONType::String ||
                        !pwl_tables[i].hasKey("dtype") ||
                        pwl_tables[i]["dtype"].valueType() != json::JSONType::String ||
                        !pwl_tables[i].hasKey("range") || pwl_tables[i]["range"].valueType() != json::JSONType::Array ||
                        !pwl_tables[i].hasKey("shift") || pwl_tables[i]["shift"].valueType() != json::JSONType::Array ||
                        !pwl_tables[i].hasKey("bias") || pwl_tables[i]["bias"].valueType() != json::JSONType::Array ||
                        !pwl_tables[i].hasKey("float_min") ||
                        pwl_tables[i]["float_min"].valueType() != json::JSONType::NumberFloat ||
                        !pwl_tables[i].hasKey("float_max") ||
                        pwl_tables[i]["float_max"].valueType() != json::JSONType::NumberFloat ||
                        !pwl_tables[i].hasKey("post_shift") ||
                        pwl_tables[i]["post_shift"].valueType() != json::JSONType::NumberInteger) {
                        reset();
                        return false;
                    }

                    auto range = pwl_tables[i]["range"];
                    std::vector<int> vecRange;
                    for (size_t ri = 0; ri < range.size(); ++ri) {
                        if (range[ri].valueType() != json::JSONType::NumberInteger) {
                            reset();
                            return false;
                        }
                        vecRange.push_back(static_cast<int>(range[ri].get<long long>()));
                    }

                    auto shift = pwl_tables[i]["shift"];
                    std::vector<int> vecShift;
                    for (size_t si = 0; si < shift.size(); ++si) {
                        if (shift[si].valueType() != json::JSONType::NumberInteger) {
                            reset();
                            return false;
                        }
                        vecShift.push_back(static_cast<int>(shift[si].get<long long>()));
                    }

                    auto bias = pwl_tables[i]["bias"];
                    std::vector<int> vecBias;
                    for (size_t bi = 0; bi < bias.size(); ++bi) {
                        if (bias[bi].valueType() != json::JSONType::NumberInteger) {
                            reset();
                            return false;
                        }
                        vecBias.push_back(static_cast<int>(bias[bi].get<long long>()));
                    }

                    PWLTableType pwl_type;
                    pwl_type.activation = pwl_tables[i]["activation"].get<std::string>();
                    pwl_type.dtype = mv::DType(pwl_tables[i]["dtype"].get<std::string>());

                    int post_shift = static_cast<int>(pwl_tables[i]["post_shift"].get<long long>());

                    double range_min = pwl_tables[i]["float_min"].get<double>();
                    double range_max = pwl_tables[i]["float_max"].get<double>();

                    PWLTableEntry pwl_table = {vecRange, vecShift, vecBias, std::make_pair(range_min, range_max),
                                               post_shift};

                    if (generalConfigs_.pwlTables.find(pwl_type) != generalConfigs_.pwlTables.end()) {
                        generalConfigs_.pwlTables.at(pwl_type).push_back(pwl_table);
                    } else {
                        std::vector<PWLTableEntry> vecTable;
                        vecTable.push_back(pwl_table);
                        generalConfigs_.pwlTables.insert({pwl_type, vecTable});
                    }
                }
            }
        }
    }

    if (jsonDescriptor["dtype"].valueType() != json::JSONType::Object)
    {
        reset();
        return false;
    }
    else
    {
        if (!jsonDescriptor["dtype"].hasKey("global"))
        {
            reset();
            return false;
        }
        else
        {
            globalDType_ = DType(jsonDescriptor["dtype"]["global"].get<std::string>());
        }

        if (!jsonDescriptor["dtype"].hasKey("hwCompatibilityCases") ||
            jsonDescriptor["dtype"]["hwCompatibilityCases"].valueType() != json::JSONType::Array)
        {
            reset();
            return false;
        }
        else
        {
            auto cases = jsonDescriptor["dtype"]["hwCompatibilityCases"];
            for (size_t cIdx = 0; cIdx < cases.size(); ++cIdx)
            {
                if (!cases[cIdx].hasKey("failCase") ||
                    cases[cIdx]["failCase"].valueType() != json::JSONType::Array ||
                    !cases[cIdx].hasKey("mitigation") ||
                    cases[cIdx]["mitigation"].valueType() != json::JSONType::Array ||
                    cases[cIdx]["failCase"].size() != cases[cIdx]["mitigation"].size())
                {
                    reset();
                    return false;
                }

                auto failCaseJson = cases[cIdx]["failCase"];
                auto mitigationJson = cases[cIdx]["mitigation"];
                DataTypeSupport dtypeSupportCase;
                for (size_t fIdx = 0; fIdx < failCaseJson.size(); ++fIdx)
                {
                    if (!failCaseJson[fIdx].hasKey("tensor") ||
                        !failCaseJson[fIdx].hasKey("dtype") ||
                        !mitigationJson[fIdx].hasKey("tensor") ||
                        !mitigationJson[fIdx].hasKey("dtype"))
                    {
                        reset();
                        return false;
                    }

                    dtypeSupportCase.failCase.push_back(
                        std::make_pair(
                            failCaseJson[fIdx]["tensor"].get<std::string>(),
                            failCaseJson[fIdx]["dtype"].get<std::string>()));
                    dtypeSupportCase.mitigation.push_back(
                        std::make_pair(
                            mitigationJson[fIdx]["tensor"].get<std::string>(),
                            mitigationJson[fIdx]["dtype"].get<std::string>()));

                }
                dtypeSupport_.push_back(dtypeSupportCase);
            }
        }
    }

    if (jsonDescriptor["resources"].valueType() != json::JSONType::Object)
    {
        reset();
        return false;
    }
    else
    {
        if (jsonDescriptor["resources"]["memory"].valueType() != json::JSONType::Array)
        {
            reset();
            return false;
        }
        else
        {

            for (std::size_t i = 0; i < jsonDescriptor["resources"]["memory"].size(); ++i)
            {

                std::string name;
                long long size;
                std::size_t alignment;

                if (!jsonDescriptor["resources"]["memory"][i].hasKey("name") ||
                    !jsonDescriptor["resources"]["memory"][i].hasKey("size") ||
                    !jsonDescriptor["resources"]["memory"][i].hasKey("alignment"))
                {
                    reset();
                    return false;
                }

                if (jsonDescriptor["resources"]["memory"][i]["name"].valueType() != json::JSONType::String ||
                    jsonDescriptor["resources"]["memory"][i]["size"].valueType() != json::JSONType::NumberInteger ||
                    jsonDescriptor["resources"]["memory"][i]["alignment"].valueType() != json::JSONType::NumberInteger)
                {
                    reset();
                    return false;
                }

                name = jsonDescriptor["resources"]["memory"][i]["name"].get<std::string>();
                size = jsonDescriptor["resources"]["memory"][i]["size"].get<long long>();
                alignment = jsonDescriptor["resources"]["memory"][i]["alignment"].get<long long>();

                // TODO
                // is it required as comparison of unsigned expression < 0 is always false
                if (size < 0)
                {
                    reset();
                    return false;
                }

                memoryDefs_[name] = {size, alignment};

            }

        }

        if (jsonDescriptor["resources"].hasKey("nce_block"))
        {
            if (jsonDescriptor["resources"]["nce_block"].valueType() != json::JSONType::Array)
            {
                reset();
                return false;
            }
            else
            {

                for (std::size_t i = 0; i < jsonDescriptor["resources"]["nce_block"].size(); ++i)
                {
                    if (!jsonDescriptor["resources"]["nce_block"][i].hasKey("name") ||
                        !jsonDescriptor["resources"]["nce_block"][i].hasKey("totalNumber"))
                    {
                        reset();
                        return false;
                    }

                    if (jsonDescriptor["resources"]["nce_block"][i]["name"].valueType() != json::JSONType::String ||
                        jsonDescriptor["resources"]["nce_block"][i]["totalNumber"].valueType() != json::JSONType::NumberInteger)
                    {
                        reset();
                        return false;
                    }

                    std::string name = jsonDescriptor["resources"]["nce_block"][i]["name"].get<std::string>();
                    long long totalNumber = jsonDescriptor["resources"]["nce_block"][i]["totalNumber"].get<long long>();

                    if (totalNumber < 0)
                    {
                        reset();
                        return false;
                    }

                    nceDefs_[name] = {static_cast<std::size_t>(totalNumber)};

                }

            }
        }

        if (jsonDescriptor["resources"].hasKey("processors"))
        {
            if (jsonDescriptor["resources"]["processors"].valueType() != json::JSONType::Array)
            {
                reset();
                return false;
            }
            else
            {

                for (std::size_t i = 0; i < jsonDescriptor["resources"]["processors"].size(); ++i)
                {
                    if (!jsonDescriptor["resources"]["processors"][i].hasKey("name") ||
                        !jsonDescriptor["resources"]["processors"][i].hasKey("totalNumber"))
                    {
                        reset();
                        return false;
                    }

                    if (jsonDescriptor["resources"]["processors"][i]["name"].valueType() != json::JSONType::String ||
                        jsonDescriptor["resources"]["processors"][i]["totalNumber"].valueType() != json::JSONType::NumberInteger)
                    {
                        reset();
                        return false;
                    }

                    std::string name = jsonDescriptor["resources"]["processors"][i]["name"].get<std::string>();
                    long long totalNumber = jsonDescriptor["resources"]["processors"][i]["totalNumber"].get<long long>();

                    if (totalNumber < 0)
                    {
                        reset();
                        return false;
                    }

                    processorDefs_[name] = {static_cast<std::size_t>(totalNumber)};


                }

            }
        }
        if (jsonDescriptor.hasKey("workloads"))
        {
            if (!jsonDescriptor["workloads"].hasKey("General"))
            {
                reset();
                return false;
            }

            std::vector<std::string> keys = jsonDescriptor["workloads"].getKeys();
            for (std::size_t l = 0; l < keys.size(); ++l)
            {
                std::string opStr = keys.at(l);
                WorkloadConfig newConfig;

                if (jsonDescriptor["workloads"][opStr].hasKey("mpe_modes"))
                {
                    if (jsonDescriptor["workloads"][opStr]["mpe_modes"].valueType() != json::JSONType::Array)
                    {
                        reset();
                        return false;
                    }
                    else
                    {
                        for (std::size_t i = 0; i < jsonDescriptor["workloads"][opStr]["mpe_modes"].size(); ++i)
                        {
                            if (jsonDescriptor["workloads"][opStr]["mpe_modes"][i].valueType() != json::JSONType::Array ||
                                jsonDescriptor["workloads"][opStr]["mpe_modes"][i].size() != 2 ||
                                jsonDescriptor["workloads"][opStr]["mpe_modes"][i][0].valueType() != json::JSONType::NumberInteger ||
                                jsonDescriptor["workloads"][opStr]["mpe_modes"][i][1].valueType() != json::JSONType::NumberInteger)
                            {
                                reset();
                                return false;
                            }
                            else
                            {
                                DPUMode newMode;
                                newMode.H = jsonDescriptor["workloads"][opStr]["mpe_modes"][i][0].get<long long>();
                                newMode.W = jsonDescriptor["workloads"][opStr]["mpe_modes"][i][1].get<long long>();
                                newConfig.dpuModes.push_back(newMode);
                            }
                        }

                    }
                }
                if (jsonDescriptor["workloads"][opStr].hasKey("algorithms"))
                {
                    if (jsonDescriptor["workloads"][opStr]["algorithms"].valueType() != json::JSONType::Array)
                    {
                        reset();
                        return false;
                    }
                    else
                    {
                        for (std::size_t i = 0; i < jsonDescriptor["workloads"][opStr]["algorithms"].size(); ++i)
                        {
                            if (jsonDescriptor["workloads"][opStr]["algorithms"][i].valueType() != json::JSONType::String)
                            {
                                reset();
                                return false;
                            }
                            else
                            {
                                newConfig.algorithms.push_back(jsonDescriptor["workloads"][opStr]["algorithms"][i].get<std::string>());
                            }
                        }

                    }
                }
                if (jsonDescriptor["workloads"][opStr].hasKey("valid_ztiling"))
                {
                    if (jsonDescriptor["workloads"][opStr]["valid_ztiling"].valueType() != json::JSONType::Array)
                    {
                        reset();
                        return false;
                    }
                    else
                    {
                        for (std::size_t i = 0; i < jsonDescriptor["workloads"][opStr]["valid_ztiling"].size(); ++i)
                        {
                            if (jsonDescriptor["workloads"][opStr]["valid_ztiling"][i].valueType() != json::JSONType::NumberInteger)
                            {
                                reset();
                                return false;
                            }
                            else
                            {
                                newConfig.validZTiles.push_back(jsonDescriptor["workloads"][opStr]["valid_ztiling"][i].get<long long>());
                            }
                        }

                    }
                }
                workloadConfigs_[opStr] = newConfig;
            }
        }
        if (jsonDescriptor["resources"]["huffman_decode_engine"].valueType() == json::JSONType::Array
            && jsonDescriptor["resources"]["bitcompactor"].valueType() == json::JSONType::Array)
        {
            reset();
            return false;
        }
        else
        {

            if (jsonDescriptor["target"].get<std::string>() == "ma2490" ||
                jsonDescriptor["target"].get<std::string>() == "ma3100" ||
                jsonDescriptor["target"].get<std::string>() == "ma3720")
            {
                for (std::size_t i = 0; i < jsonDescriptor["resources"]["huffman_decode_engine"].size(); ++i)
                {

                    std::string name;
                    std::size_t numberOfHDEModules;
                    std::size_t bitPerSymbol;
                    std::size_t blockSize;
                    std::size_t maxNumberEncodedSymbols;
                    bool bypassMode;

                    if (!jsonDescriptor["resources"]["huffman_decode_engine"][i].hasKey("name") ||
                        !jsonDescriptor["resources"]["huffman_decode_engine"][i].hasKey("numberOfHDEModules") ||
                        !jsonDescriptor["resources"]["huffman_decode_engine"][i].hasKey("bitPerSymbol") ||
                        !jsonDescriptor["resources"]["huffman_decode_engine"][i].hasKey("blockSize") ||
                        !jsonDescriptor["resources"]["huffman_decode_engine"][i].hasKey("maxNumberEncodedSymbols") ||
                        !jsonDescriptor["resources"]["huffman_decode_engine"][i].hasKey("bypassMode"))
                    {
                        reset();
                        return false;
                    }

                    if (jsonDescriptor["resources"]["huffman_decode_engine"][i]["name"].valueType() != json::JSONType::String ||
                        jsonDescriptor["resources"]["huffman_decode_engine"][i]["numberOfHDEModules"].valueType() != json::JSONType::NumberInteger ||
                        jsonDescriptor["resources"]["huffman_decode_engine"][i]["bitPerSymbol"].valueType() != json::JSONType::NumberInteger ||
                        jsonDescriptor["resources"]["huffman_decode_engine"][i]["blockSize"].valueType() != json::JSONType::NumberInteger ||
                        jsonDescriptor["resources"]["huffman_decode_engine"][i]["maxNumberEncodedSymbols"].valueType() != json::JSONType::NumberInteger ||
                        jsonDescriptor["resources"]["huffman_decode_engine"][i]["bypassMode"].valueType() != json::JSONType::Bool)
                    {
                        reset();
                        return false;
                    }

                    name = jsonDescriptor["resources"]["huffman_decode_engine"][i]["name"].get<std::string>();
                    numberOfHDEModules = jsonDescriptor["resources"]["huffman_decode_engine"][i]["numberOfHDEModules"].get<long long>();
                    bitPerSymbol = jsonDescriptor["resources"]["huffman_decode_engine"][i]["bitPerSymbol"].get<long long>();
                    blockSize = jsonDescriptor["resources"]["huffman_decode_engine"][i]["blockSize"].get<long long>();
                    maxNumberEncodedSymbols = jsonDescriptor["resources"]["huffman_decode_engine"][i]["maxNumberEncodedSymbols"].get<long long>();
                    bypassMode = jsonDescriptor["resources"]["huffman_decode_engine"][i]["bypassMode"].get<bool>();

                    // TODO
                    // is it required as comparison of unsigned expression < 0 is always false
                    if (std::less<size_t>()(numberOfHDEModules, 0U))
                    {
                        reset();
                        return false;
                    }

                    auto hdeDef = std::make_shared<HdeDescriptor>();

                    hdeDef->numberOfHDEModules = numberOfHDEModules;
                    hdeDef->bitPerSymbol = bitPerSymbol;
                    hdeDef->blockSize = blockSize;
                    hdeDef->maxNumberEncodedSymbols = maxNumberEncodedSymbols;
                    hdeDef->bypassMode = bypassMode;
                    hdeDef->codecName = mv::CodecType::HDE;

                    codecDef_ = hdeDef;
                }
            }

            else if (jsonDescriptor["target"].get<std::string>() == "ma3720")
            {
                for (std::size_t i = 0; i < jsonDescriptor["resources"]["bitcompactor"].size(); ++i)
                {

                    std::string name;
                    std::size_t numCompressionModules;
                    std::size_t bufferAlignment;
                    std::size_t bitmapPreprocEnable;
                    bool bypassMode;


                    if (!jsonDescriptor["resources"]["bitcompactor"][i].hasKey("name") ||
                        !jsonDescriptor["resources"]["bitcompactor"][i].hasKey("numCompressionModules") ||
                        !jsonDescriptor["resources"]["bitcompactor"][i].hasKey("bufferAlignment") ||
                        !jsonDescriptor["resources"]["bitcompactor"][i].hasKey("bitmapPreprocEnable") ||
                        !jsonDescriptor["resources"]["bitcompactor"][i].hasKey("bypassMode"))
                    {
                        reset();
                        return false;
                    }

                    if (jsonDescriptor["resources"]["bitcompactor"][i]["name"].valueType() != json::JSONType::String ||
                        jsonDescriptor["resources"]["bitcompactor"][i]["numCompressionModules"].valueType() != json::JSONType::NumberInteger ||
                        jsonDescriptor["resources"]["bitcompactor"][i]["bufferAlignment"].valueType() != json::JSONType::NumberInteger ||
                        jsonDescriptor["resources"]["bitcompactor"][i]["bitmapPreprocEnable"].valueType() != json::JSONType::NumberInteger ||
                        jsonDescriptor["resources"]["bitcompactor"][i]["bypassMode"].valueType() != json::JSONType::Bool)
                    {
                        reset();
                        return false;
                    }

                    name = jsonDescriptor["resources"]["bitcompactor"][i]["name"].get<std::string>();
                    numCompressionModules = jsonDescriptor["resources"]["bitcompactor"][i]["numCompressionModules"].get<long long>();
                    bufferAlignment = jsonDescriptor["resources"]["bitcompactor"][i]["bufferAlignment"].get<long long>();
                    bitmapPreprocEnable = jsonDescriptor["resources"]["bitcompactor"][i]["bitmapPreprocEnable"].get<long long>();
                    bypassMode = jsonDescriptor["resources"]["bitcompactor"][i]["bypassMode"].get<bool>();

                    if (std::less<size_t>()(numCompressionModules, 0U))
                    {
                        reset();
                        return false;
                    }

                    auto btcDef = std::make_shared<BTCDescriptor>();

                    btcDef->numCompressionModules = numCompressionModules;
                    btcDef->bufferAlignment = bufferAlignment;
                    btcDef->bitmapPreprocEnable = bitmapPreprocEnable;
                    btcDef->bypassMode = bypassMode;
                    btcDef->codecName = mv::CodecType::BTC;

                    codecDef_ = btcDef;

                }
            }

        }

    }

    return true;

}

bool mv::TargetDescriptor::save(const std::string& filePath)
{

    // open a file in read mode.
    std::ofstream descFile;
    descFile.open(filePath, std::ios::out | std::ios::trunc);

    if (!descFile.is_open())
        return false;

    json::Object root;
    root["target"] = toString(target_);
    root["dtype"] = globalDType_.toString();

    descFile << root.stringifyPretty();
    descFile.close();

    return true;

}

void mv::TargetDescriptor::setTarget(Target target)
{
    if (target == Target::Unknown)
        throw ArgumentError(*this, "target", "unknown", "Defining target as unknown is illegal");
    target_ = target;
}

void mv::TargetDescriptor::setDType(DType dType)
{
    globalDType_ = dType;
}


bool mv::TargetDescriptor::defineMemory(const std::string& name, long long size, std::size_t alignment)
{
    if (size < 0)
        return false;

    if (memoryDefs_.find(name) == memoryDefs_.end())
    {
        memoryDefs_[name] = {size, alignment};
        return true;
    }
    return false;
}

bool mv::TargetDescriptor::undefineMemory(const std::string& name)
{
    auto defIt = memoryDefs_.find(name);
    if (defIt != memoryDefs_.end())
    {
        memoryDefs_.erase(defIt);
        return true;
    }
    return false;
}

mv::Target mv::TargetDescriptor::getTarget() const
{
    return target_;
}

mv::DType mv::TargetDescriptor::getDType() const
{
    return globalDType_;
}

mv::CodecType mv::TargetDescriptor::getCodecName() const
{
    return codecDef_->codecName;
}

const std::map<std::string, mv::TargetDescriptor::MemoryDescriptor>& mv::TargetDescriptor::memoryDefs() const
{
    return memoryDefs_;
}

const std::map<std::string, mv::TargetDescriptor::NceDescriptor>& mv::TargetDescriptor::nceDefs() const
{
    return nceDefs_;
}

const std::shared_ptr<mv::CodecDescriptor>& mv::TargetDescriptor::codecDef() const
{
    return codecDef_;
}

const std::vector<mv::DataTypeSupport>& mv::TargetDescriptor::dtypeSupport() const
{
    return dtypeSupport_;
}

const std::map<std::string, std::size_t>& mv::TargetDescriptor::processorDefs() const
{
    return processorDefs_;
}

const std::map<std::string, mv::WorkloadConfig>& mv::TargetDescriptor::getWorkloadConfigs() const
{
    return workloadConfigs_;
}

std::string mv::TargetDescriptor::getLogID() const
{
    return "TargetDescriptor";
}

const mv::GeneralTargetConfigs& mv::TargetDescriptor::generalTargetConfigs() const
{
    return generalConfigs_;
}
