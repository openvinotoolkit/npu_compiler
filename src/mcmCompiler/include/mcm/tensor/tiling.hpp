//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifndef TILING_HPP_
#define TILING_HPP_

#include "include/mcm/base/element.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"

namespace mv
{
    class Tiling : public LogSender {
    private:
        using TileShape = std::vector<std::size_t>;

        enum TileDim { TILE_DIM_W, TILE_DIM_H, TILE_DIM_C, TILE_DIM_K, TILE_DIM_N };
        TileShape start_;
        TileShape size_;

        std::string axis_;
        std::vector<Tiling> childTiles_;

        std::size_t alignment_;

        void printShape(const TileShape& shape) const
        {
            std::cout<< "{";
            for(size_t i = 0; i <= TILE_DIM_N; ++i)
                std::cout<< shape[i] << "," ;
            std::cout<<"}";
        }

        virtual std::string getLogID() const override
        {
            return "Tilling";
        }

    public:

        Tiling() : start_({0,0,0,0,0}), size_({0,0,0,0,0}),
                axis_(""), childTiles_(0), alignment_(1) {}
        Tiling(const Shape& actShape, const Shape& kernelShape)
                : start_({0,0,0,0,0}), axis_(""), childTiles_(0), alignment_(1)
        {
            size_.resize(5);
            size_[TILE_DIM_W] = actShape[mv::IO_WIDTH_DIMENSION];
            size_[TILE_DIM_H] = actShape[mv::IO_HEIGHT_DIMENSION];
            size_[TILE_DIM_C] = actShape[mv::IO_CHANNEL_DIMENSION];
            size_[TILE_DIM_K] = kernelShape[mv::KERNEL_OUTPUT_CHANNELS];
            size_[TILE_DIM_N] = actShape[mv::IO_BATCH_DIMENSION];
        }
        Tiling(const Shape& actShape)
                : start_({0,0,0,0,0}),axis_(""), childTiles_(0), alignment_(1)
        {
            size_.resize(5);
            size_[TILE_DIM_W] = actShape[mv::IO_WIDTH_DIMENSION];
            size_[TILE_DIM_H] = actShape[mv::IO_HEIGHT_DIMENSION];
            size_[TILE_DIM_C] = actShape[mv::IO_CHANNEL_DIMENSION];
            size_[TILE_DIM_K] = actShape[mv::IO_CHANNEL_DIMENSION];
            size_[TILE_DIM_N] = actShape[mv::IO_BATCH_DIMENSION];
        }

        Tiling(const TileShape& start, const TileShape& size) :
                start_(start),size_(size),axis_(""),childTiles_(0), alignment_(1) {}

        Tiling(const std::string& axis, std::size_t tiles)
                : start_({0,0,0,0,0}), size_({0,0,0,0,0}),
                axis_(axis), childTiles_(tiles), alignment_(1)
        {
        }

        Tiling(const Shape& start, Shape& size, const std::string& axis, std::size_t childTiles)
                : start_(start), size_(size), axis_(axis), childTiles_(childTiles),
                alignment_(1)
        {
        }

        Tiling& operator=(const Tiling& other)
        {
            start_= other.start_;
            size_ = other.size_;
            axis_ = other.axis_;
            childTiles_ = other.childTiles_;
            alignment_ = other.alignment_;
            return *this;
        }

        Tiling(const Tiling&) = default;

        const std::string& getAxis() const { return axis_; }
        void setAxis(const std::string& axis) { axis_ = axis; }

        std::size_t getAlignment() { return alignment_; }
        void setAlignment(std::size_t alignment) { alignment_ = alignment; }

        const TileShape& getStartCoord() const { return start_; }
        void setStartCoord(const TileShape& start) { start_ = start; }

        const TileShape& getSize() const { return size_; }
        void setSize(const TileShape& size) { size_ = size; }

        // TODO: This method requires const correctness, but it can't be changed because
        // spatial_split_streaming has already started using it. Will be replaced by
        // getChildTiles().
        std::vector<Tiling>& childTiles() { return childTiles_; }
        const std::vector<Tiling>& getChildTiles() const { return childTiles_; }
        void setChildTile(const Tiling& tile, std::size_t index) { childTiles_[index] = tile; }

        void resizeNumberOfTiles(std::size_t children) { childTiles_.resize(children); }

        mv::Shape getActivationShape()
        {
            return mv::Shape({size_[TILE_DIM_W],size_[TILE_DIM_H],size_[TILE_DIM_C],size_[TILE_DIM_N]});
        }
        mv::Shape getActivationStart()
        {
            return mv::Shape({start_[TILE_DIM_W],start_[TILE_DIM_H],start_[TILE_DIM_C],start_[TILE_DIM_N]});
        }
        mv::Shape getKernelShape()
        {
            return mv::Shape({0,0,size_[TILE_DIM_C],size_[TILE_DIM_K]});
        }
        mv::Shape getKernelStart()
        {
            return mv::Shape({0,0,start_[TILE_DIM_C],start_[TILE_DIM_K]});
        }

        static inline std::size_t inferInputSize(std::size_t outputSize, std::size_t padding_start, std::size_t padding_end, std::size_t kernel_size, std::size_t kernel_stride)
        {
            const std::size_t inputSize = ((outputSize - 1) * kernel_stride) - padding_start - padding_end + kernel_size;
            return inputSize;
        }

        static inline std::size_t inferOutputSize(std::size_t inputSize, std::size_t padding_start, std::size_t padding_end, std::size_t kernel_size, std::size_t kernel_stride)
        {
            const std::size_t outputSize = ( inputSize + padding_start + padding_end - kernel_size) / kernel_stride + 1;
            return outputSize;
        }

        void generateWeightsTiling()
        {
            auto numberOfSplits = childTiles_.size();
            auto parentTileShape = getSize();
            auto axisToSplit = mv::Shape::getAxis(getAxis());
            int newSize = ceil(((double)parentTileShape[axisToSplit]) / ((double)numberOfSplits));
            newSize = round_up(newSize, alignment_);
            size_t remainderSize = parentTileShape[axisToSplit] - (newSize*(numberOfSplits -1));

            if(remainderSize == 0)
            {
                //this means that whoever gave the NR of streams did not take into account that channels need to be rounded
                numberOfSplits--;
                childTiles_.pop_back();
                remainderSize = newSize;
                this->log(mv::Logger::MessageType::Warning, "Zero remainder size, subtracting number of splits to " +
                    std::to_string(numberOfSplits) + " each of size " + std::to_string(newSize));
            }

            unsigned startCoord = 0;


            for(std::size_t split = 0; split < numberOfSplits; split++)
            {
                TileShape tileStart({0,0,0,0,0});
                TileShape tileSize = parentTileShape;

                tileStart[axisToSplit] = startCoord;
                startCoord += newSize;
                if(split == (numberOfSplits-1))
                    tileSize[axisToSplit] = remainderSize;
                else
                    tileSize[axisToSplit] = newSize;
                mv::Tiling newTile(tileStart, tileSize);
                setChildTile(newTile, split);
            }
        }

        void computeTiling(mv::Data::OpListIterator opIt, const std::vector<mv::Shape>& outputTileSizes, const std::vector<mv::Shape>& outputTileStarts)
        {
            auto numberOfSplits = childTiles().size();
            auto inputShape = getSize();
            auto axisToSplit =  mv::Shape::getAxis(getAxis());

            size_t kernelSize;
            std::string opType = opIt->getOpType();
            if (opType == "Conv" || opType == "DepthwiseConv")
            {
                auto weightTensor = opIt->getInputTensor(1);
                auto weightsShape = weightTensor->getShape();
                auto kernelDin = (axisToSplit == mv::Shape::getAxis("W")) ? mv::KERNEL_WIDTH : mv::KERNEL_HEIGHT;
                kernelSize = weightsShape[kernelDin];
            }
            else
            {
                if (opIt->hasAttr("kSize"))
                    kernelSize = opIt->get<std::array<unsigned short, 2UL>>("kSize")[0];
                else
                    kernelSize = 1;//fake kernel
            }

            //todo:: is there any macro for kernel w/h order?
            auto kernelAxis = (axisToSplit == mv::Shape::getAxis("W")) ? 0 : 1;
            unsigned short kernelStride;
            if (opIt->hasAttr("stride"))
                kernelStride = opIt->get<std::array<unsigned short, 2>>("stride")[kernelAxis];
            else
                kernelStride = 1;//fake stride
            std::array<unsigned short, 4> padding;
            if (opIt->hasAttr("padding"))
                padding = opIt->get<std::array<unsigned short, 4>>("padding");
            else
                padding = {0,0,0,0};

            int padStart=0, padEnd=0;

            if (axisToSplit == mv::Shape::getAxis("H"))
            {
                padStart = padding[mv::PADDING_TOP];
                padEnd = padding[mv::PADDING_BOT];
            }

            //NOTE: the idea below is that we will try to catch the cases with brute force !!! cause a
            // general rule seems over-complicated to apply
            size_t startCoord = 0;
            //case-A: kernel = stride and no pads: input tiles == output tiles
            //case-B: OutputTiles are overlapping
            //case-C: like usual Streaming, were the output tiles are not overlapping
            bool equalInputOutput = (kernelSize == kernelStride);
            bool zeroPads = (padStart == 0) && (padEnd == 0);
            bool overlappingOutput = false;
            if (!(equalInputOutput && zeroPads))
            {
                for (std::size_t split = 1; split < numberOfSplits; ++split)
                {
                    auto tilePreviousSize = outputTileSizes[split - 1][mv::IO_HEIGHT_DIMENSION];
                    auto tileCurrentStart = outputTileStarts[split][mv::IO_HEIGHT_DIMENSION];
                    if (tileCurrentStart < tilePreviousSize)
                    {
                        overlappingOutput = true;
                        break;
                    }
                }
            }

            if (equalInputOutput && zeroPads)
            {
                for (std::size_t split = 0; split < numberOfSplits; ++split)
                {
                    TileShape tileStart({0,0,0,0,0});
                    TileShape tileSize = inputShape;

                    auto outputHeightDim = outputTileSizes[split][mv::IO_HEIGHT_DIMENSION];
                    tileStart[axisToSplit] = outputTileStarts[split][mv::IO_HEIGHT_DIMENSION];
                    tileSize[axisToSplit] = outputHeightDim;

                    mv::Tiling newTile(tileStart, tileSize);
                    setChildTile(newTile, split);
                }
            }
            else if (overlappingOutput)
            {
                // Input tiling must start from 0. Shift the index when the first start_index is not 0.
                int64_t biasStart = 0;
                for (std::size_t split = 0; split < numberOfSplits; ++split)
                {
                    TileShape tileStart({0,0,0,0,0});
                    TileShape tileSize = inputShape;
                    auto outputHeightDim = outputTileSizes[split][mv::IO_HEIGHT_DIMENSION];
                    int64_t tileStarting = inferInputSize(outputTileStarts[split][mv::IO_HEIGHT_DIMENSION],padStart,padEnd,kernelSize,kernelStride) - 1;

                    if (split == 0 && tileStarting > 0)
                        biasStart = tileStarting;
                    if (tileStarting < -1)
                        throw mv::RuntimeError("Tiling", "Negative Tile begins!");
                    else if (tileStarting == -1)
                        tileStarting = 0;

                    tileStart[axisToSplit] = tileStarting - biasStart;

                    if (split == 0)
                        tileSize[axisToSplit] = inferInputSize(outputHeightDim,padStart,0,kernelSize,kernelStride);
                    else if (split == (numberOfSplits-1))
                        tileSize[axisToSplit] = inferInputSize(outputHeightDim,0,padEnd,kernelSize,kernelStride);
                    else
                        tileSize[axisToSplit] = inferInputSize(outputHeightDim,0,0,kernelSize,kernelStride);

                    mv::Tiling newTile(tileStart, tileSize);
                    setChildTile(newTile, split);
                }
            }
            else
            {
                for (std::size_t split = 0; split < numberOfSplits; ++split)
                {
                    TileShape tileStart({0,0,0,0,0});
                    TileShape tileSize = inputShape;
                    auto outputHeightDim = outputTileSizes[split][mv::IO_HEIGHT_DIMENSION];

                    tileStart[axisToSplit] = startCoord;

                    if (split == 0)
                        tileSize[axisToSplit] = inferInputSize(outputHeightDim,padStart,0,kernelSize,kernelStride);
                    else if (split == (numberOfSplits-1))
                    {
                        tileSize[axisToSplit] = inferInputSize(outputHeightDim,0,padEnd,kernelSize,kernelStride);
                        while (tileStart[axisToSplit] + tileSize[axisToSplit] > inputShape[axisToSplit])
                            tileStart[axisToSplit] = tileStart[axisToSplit] - 1;
                    }
                    else
                        tileSize[axisToSplit] = inferInputSize(outputHeightDim,0,0,kernelSize,kernelStride);


                    mv::Tiling newTile(tileStart, tileSize);
                    setChildTile(newTile, split);

                    if (split == 0)
                        startCoord += outputHeightDim * kernelStride - (inferInputSize(outputHeightDim,0,0,kernelSize,kernelStride) - tileSize[axisToSplit]);
                    else
                        startCoord += outputHeightDim * kernelStride;
                }
            }
        }

        void generateSpatialTiling(mv::Data::OpListIterator opIt)
        {
            auto numberOfSplits = childTiles().size();
            auto inputShape = getSize();
            auto axisToSplit =  mv::Shape::getAxis(getAxis());

            size_t kernelSize;
            std::string opType = opIt->getOpType();
            if (opType == "Conv" || opType == "DepthwiseConv")
            {
                auto weightTensor = opIt->getInputTensor(1);
                auto weightsShape = weightTensor->getShape();
                auto kernelDin = (axisToSplit == mv::Shape::getAxis("W")) ? mv::KERNEL_WIDTH : mv::KERNEL_HEIGHT;
                kernelSize = weightsShape[kernelDin];
            }
            else
            {
                if (opIt->hasAttr("kSize"))
                    kernelSize = opIt->get<std::array<unsigned short, 2UL>>("kSize")[0];
                else
                    kernelSize = 1;//fake kernel
            }

            //todo:: is there any macro for kernel w/h order?
            auto kernelAxis = (axisToSplit == mv::Shape::getAxis("W")) ? 0 : 1;
            unsigned short kernelStride;
            if (opIt->hasAttr("stride"))
                kernelStride = opIt->get<std::array<unsigned short, 2>>("stride")[kernelAxis];
            else
                kernelStride = 1;//fake stride
            std::array<unsigned short, 4> padding;
            if (opIt->hasAttr("padding"))
                padding = opIt->get<std::array<unsigned short, 4>>("padding");
            else
                padding = {0,0,0,0};

            int padStart=0,padEnd=0;

            if (axisToSplit == mv::Shape::getAxis("W"))
            {
                padStart = padding[0];
                padEnd = padding[1];
            }
            else if (axisToSplit == mv::Shape::getAxis("H"))
            {
                padStart = padding[2];
                padEnd = padding[3];
            }

            int outputSize =  inferOutputSize(inputShape[axisToSplit],padStart,padEnd,kernelSize,kernelStride);
            auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);

            size_t startCoord = 0;
            int curPadStart = padStart;
            int curPadEnd = 0;
            for (std::size_t split = 0; split < numberOfSplits; split++)
            {
                TileShape tileStart({0,0,0,0,0});
                TileShape tileSize = inputShape;

                tileStart[axisToSplit] = startCoord;

                tileSize[axisToSplit] = inferInputSize(newOutputSizes[split],curPadStart,curPadEnd,kernelSize,kernelStride);

                // check end padding and re-infer
                if (startCoord + tileSize[axisToSplit] > inputShape[axisToSplit])
                {
                    curPadEnd += startCoord + tileSize[axisToSplit] - inputShape[axisToSplit];
                    tileSize[axisToSplit] = inferInputSize(newOutputSizes[split],curPadStart,curPadEnd,kernelSize,kernelStride);
                }

                mv::Tiling newTile(tileStart, tileSize);
                setChildTile(newTile, split);

                auto noPadStartCoord = newOutputSizes[split] * kernelStride;
                auto padBias = inferInputSize(newOutputSizes[split], 0, 0, kernelSize, kernelStride) - tileSize[axisToSplit];
                // check startCoord and start padding
                if (startCoord == 0 || noPadStartCoord < padBias)
                {
                    // still need padding start
                    if (noPadStartCoord < padBias)
                    {
                        curPadStart = padBias - noPadStartCoord;
                        startCoord = 0;
                    }
                    // pad start finishes
                    else
                    {
                        startCoord += noPadStartCoord - padBias;
                        curPadStart = 0;
                    }
                }
                else
                {
                    startCoord += noPadStartCoord;
                    padStart = 0;
                }
            }
        }

        void generateBatchTiling()
        {
            auto numberOfSplits = childTiles().size();
            auto inputShape = getSize();
            // MV::Shape considers N(batches) at the same index as K(out channels for weight sets)
            // which won't work with our tiling logic where N and K sizes jave separate entries
            auto axisToSplit =  TILE_DIM_N;

            auto newInputSizes = tileSpatialOutputSize(inputShape[axisToSplit], numberOfSplits);

            size_t startCoord = 0;
            for (std::size_t split = 0; split < numberOfSplits; split++)
            {
                TileShape tileStart({0,0,0,0,0});
                TileShape tileSize = inputShape;

                tileStart[axisToSplit] = startCoord;
                tileSize[axisToSplit] = newInputSizes[split];

                mv::Tiling newTile(tileStart, tileSize);
                setChildTile(newTile, split);

                // Compute start coordinates for the next tile
                startCoord += newInputSizes[split];
            }
        }

        void generateTiling(mv::Data::OpListIterator opIt, bool verticalFusion = false, const std::vector<mv::Shape>& outputTileSizes = {}, const std::vector<mv::Shape>& outputTileStarts = {}, bool propagate = false)
        {
            if(axis_ == "K" || axis_ == "C")
                generateWeightsTiling();
            else if ((axis_ == "H" || axis_ == "W") && (!verticalFusion) && (!propagate))
                generateSpatialTiling(opIt);
            else if (axis_ == "N")
                generateBatchTiling();
            else if (verticalFusion && !propagate)
                computeTiling(opIt, outputTileSizes, outputTileStarts);
            else if (propagate && !verticalFusion)
                computeTiling(opIt, outputTileSizes, outputTileStarts);
            else
                throw mv::RuntimeError("Tiling", "Tile methodology needs to be initialized with the appropriate args!");
        }

        void print(std::ostream& o, const Tiling& tiling, int depth = 0) const {
            o << std::string(depth, '\t') << "{";
            for (std::size_t i = 0; i < tiling.getSize().size(); ++i)
            {
                if (i != 0) o << ",";
                o << tiling.getSize()[i];
            }
            o << "}" << std::endl;
            for (const auto& childTile : tiling.getChildTiles())
            {
                print(o, childTile, depth + 1);
            }
        }
    };
    // TODO: This currently leads to compile errors due to way in which the tiling.hpp
    // gets included. Uncomment this once this is resolved.
    /*
    std::ostream& operator<<(std::ostream& o, const Tiling& t) {
        t.print(o, t);
        return o;
    }
    */
}
#endif
