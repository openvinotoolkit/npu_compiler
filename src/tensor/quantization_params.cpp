#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::QuantizationParams::QuantizationParams(const json::Value& content) : Element(content)
{

}
mv::QuantizationParams::QuantizationParams(std::vector<int64_t> zp, std::vector<double> scale,
    std::vector<double> min, std::vector<double> max): Element("quantParams")
{
    size_t size = zp.size();
    if (size != scale.size() || size != min.size() || size != max.size())
        throw ArgumentError("quantParams", "Quantization params size", "",
            "Sizes of the different params don't match");

    for (size_t i = 0; i < size; i++)
        if (max[i] < min[i])
            throw ArgumentError("quantParams", "Quantization min max params", "max",
                " Smaller than min " + std::to_string(min[i]));

    set<std::vector<int64_t>>("zeroPoint", zp);
    set<std::vector<double>>("scale", scale);
    set<std::vector<double>>("min", min);
    set<std::vector<double>>("max", max);
}

mv::QuantizationParams::QuantizationParams(std::vector<int64_t> zp, std::vector<double> scale,
    std::vector<double> min, std::vector<double> max, std::vector<unsigned> shift, std::vector<unsigned> mult): Element("quantParams")
{
    size_t size = zp.size();
    if (size != scale.size() || size != min.size() || size != max.size())
        throw ArgumentError("quantParams", "Quantization params size", "",
            "Sizes of the different params don't match");

    for (size_t i = 0; i < size; i++)
        if (max[i] < min[i])
            throw ArgumentError("quantParams", "Quantization min max params", "max",
                " Smaller than min " + std::to_string(min[i]));

    set<std::vector<int64_t>>("zeroPoint", zp);
    set<std::vector<double>>("scale", scale);
    set<std::vector<double>>("min", min);
    set<std::vector<double>>("max", max);
    set<std::vector<unsigned>>("shift", shift);
    set<std::vector<unsigned>>("mult", mult);
}

void mv::QuantizationParams::quantize(std::vector<unsigned> shift, std::vector<unsigned> mult)
{
    set<std::vector<unsigned>>("shift", shift);
    set<std::vector<unsigned>>("mult", mult);
}

int64_t mv::QuantizationParams::getZeroPoint(const size_t channel) const
{
    std::vector<int64_t> zeroPoint = get<std::vector<int64_t>>("zeroPoint");
    if (zeroPoint.size() == 1)
        return zeroPoint[0];
    if (channel >= zeroPoint.size())
        throw ArgumentError("quantParams", "channel", std::to_string(channel),
            "Invalid index: channel is greater than zeroPoint vector");
    return zeroPoint[channel];
}
std::string mv::QuantizationParams::getLogID() const
{
    return "QuantizationParams: " + getName();
}

std::string mv::QuantizationParams:: toString() const
{
    return getLogID() + Element::attrsToString_();
}

bool mv::QuantizationParams:: isEmpty() const
{
    bool is_empty = false;
    if (get<std::vector<int64_t>>("zeroPoint").size() + get<std::vector<double>>("scale").size() + get<std::vector<double>>("min").size() + get<std::vector<double>>("max").size() == 0)
        is_empty = true;
    return is_empty;
}
