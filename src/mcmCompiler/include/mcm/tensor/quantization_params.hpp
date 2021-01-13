#ifndef QUANTIZATION_PARAMS_HPP_
#define QUANTIZATION_PARAMS_HPP_

#include <vector>
#include "include/mcm/base/element.hpp"
#include "include/mcm/utils/data_generator.hpp"

namespace mv
{

    class QuantizationParams: public Element
    {
    public:
        QuantizationParams(const json::Value& content);
        QuantizationParams(
            const std::vector<int64_t>& zp,
            const std::vector<double>& scale,
            const std::vector<double>& min,
            const std::vector<double>& max);
        QuantizationParams(
            const std::vector<int64_t>& zp,
            const std::vector<double>& scale,
            const std::vector<double>& min,
            const std::vector<double>& max,
            const std::vector<unsigned>& shift,
            const std::vector<unsigned>& mult);
        QuantizationParams(
            const std::vector<int64_t>& zp,
            const std::vector<double>& scale,
            const std::vector<double>& min,
            const std::vector<double>& max,
            const std::vector<unsigned>& shift,
            const std::vector<unsigned>& mult,
            const signed postShift);
//        QuantizationParams & operator=(const QuantizationParams& quantObject);

        inline std::vector<int64_t> getZeroPoint() const
        {
            return get<std::vector<int64_t>>("zeroPoint");
        }

        inline std::vector<double> getScale() const
        {
            return get<std::vector<double>>("scale");
        }

        inline std::vector<double> getMin() const
        {
            return get<std::vector<double>>("min");
        }

        inline std::vector<double> getMax() const
        {
            return get<std::vector<double>>("max");
        }

        inline std::vector<unsigned> getShift() const
        {
            return get<std::vector<unsigned>>("shift");
        }

        inline std::vector<unsigned> getMult() const
        {
            return get<std::vector<unsigned>>("mult");
        }

        inline signed getPostShift() const
        {
            return hasAttr("postShift") ? get<signed>("postShift") : 0;
        }

        void quantize(std::vector<unsigned> shift, std::vector<unsigned> mult);
        void setScale(std::vector<double> scale_);
        void setZeroPoint(std::vector<int64_t> zeroPoint_);
        void setPostShift(signed postShift_);

        int64_t getZeroPoint(const size_t channel) const;
        double getScale(const size_t channel) const;
        bool isScalePerTensor() const;

        virtual std::string getLogID() const override;
        virtual std::string toString() const override;
        virtual bool isEmpty() const;
        virtual bool isNeutral() const;
        virtual bool infinitelimits() const;

        static QuantizationParams empty();
        static QuantizationParams initial();
        // Return QuantizationParams which are created from an equally divided parts of each quantization parameter
        // in case it is a vector per channel. In case quant parameter is per tensor (vector size = 1) it is populated
        // without any change
        QuantizationParams getSlice(std::size_t slice_idx, std::size_t total_slices_number);
    };

}

#endif // QUANTIZATION_PARAMS_HPP_
