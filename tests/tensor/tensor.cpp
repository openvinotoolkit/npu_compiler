#include "gtest/gtest.h"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/order/order.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "include/mcm/tensor/binary_data.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/env_loader.hpp"

#include <fstream>

TEST(tensor, populating)
{

    mv::Shape tShape({5, 5});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(2)));
    t.populate(data);

    for (unsigned j = 0; j < tShape[0]; ++j)
        for (unsigned i = 0; i < tShape[1]; ++i)
            ASSERT_EQ((double) t({i, j}), i + tShape[0] * j);

}

/*TEST(tensor, sub_to_ind_column_major)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<std::size_t> subs[] = {
        {31, 15, 4, 2},
        {15, 7, 2, 0},
        {5, 1, 2, 3},
        {25, 5, 6, 1},
        {0, 12, 0, 3}
    };

    auto idxFcn = [tShape](const std::vector<std::size_t> &s) {
        return s[0] + tShape[0] * (s[1] + tShape[1] * (s[2] + tShape[2] * s[3]));
    };

    mv::Order mv::Order(mv::Order("CHW"));

    for (unsigned i = 0; i < 5; ++i)
        ASSERT_EQ(order.subToInd(tShape, subs[i]), idxFcn(subs[i]));



}

TEST(tensor, int_to_sub_column_major)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order("CHW"));
    t.populate(data);

    std::vector<unsigned> idx = {0, 100, 101, 545, 10663};

    for (unsigned i = 0; i < 5; ++i)
    {
        std::vector<std::size_t> sub = t.indToSub(idx[i]);
        ASSERT_EQ(t(sub), idx[i]);
    }

}

TEST(tensor, sub_to_ind_row_major)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<std::size_t> subs[] = {
        {31, 15, 4, 2},
        {15, 7, 2, 0},
        {5, 1, 2, 3},
        {25, 5, 6, 1},
        {0, 12, 0, 3}
    };

    auto idxFcn = [tShape](const std::vector<std::size_t> &s) {
        return s[3] + tShape[3] * (s[2] + tShape[2] * (s[1] + tShape[1] * s[0]));
    };

    mv::Order mv::Order(mv::Order("WHC"));

    for (unsigned i = 0; i < 5; ++i)
        ASSERT_EQ(order.subToInd(tShape, subs[i]), idxFcn(subs[i]));

}

TEST(tensor, ind_to_sub_row_major)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order("WHC"));
    t.populate(data);

    std::vector<unsigned> idx = {0, 100, 101, 545, 10663};

    for (unsigned i = 0; i < 5; ++i)
    {
        std::vector<std::size_t> sub = t.indToSub(idx[i]);
        ASSERT_EQ(t(sub), t(idx[i]));
    }

}

TEST(tensor, sub_to_ind_planar)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<std::size_t> subs[] = {
        {31, 15, 4, 2},
        {15, 7, 2, 0},
        {5, 1, 2, 3},
        {25, 5, 6, 1},
        {0, 12, 0, 3}
    };

    auto idxFcn = [tShape](const std::vector<std::size_t> &s) {
        return s[3] + tShape[3] * (s[2] + tShape[2] * (s[0] + tShape[0] * s[1]));
    };

    mv::Order mv::Order(mv::Order("HWC"));

    for (unsigned i = 0; i < 5; ++i)
        ASSERT_EQ(order.subToInd(tShape, subs[i]), idxFcn(subs[i]));

}

TEST(tensor, ind_to_sub_planar)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order("HWC"));
    t.populate(data);

    std::vector<unsigned> idx = {0, 100, 101, 545, 10663};
    auto idxFcn = [tShape](const std::vector<std::size_t>& s) {
        return s[3] + tShape[3] * (s[2] + tShape[2] * (s[0] + tShape[0] * s[1]));
    };

    for (unsigned i = 0; i < 5; ++i)
    {
        std::vector<std::size_t> sub = t.indToSub(idx[i]);
        ASSERT_EQ(t(sub), t(idx[i]));
    }

}*/

TEST(tensor, column_major_to_row_major)
{

    mv::Shape tShape({3, 3, 3, 3});
    std::vector<double> data = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f,
        65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f,
        73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)));
    t.populate(data);
    t.setOrder(mv::Order(mv::Order::getRowMajorID(4)));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ((double)t(i), reorderedData[i]);

}

TEST(tensor, row_major_to_column_major)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f,
        65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f,
        73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order(mv::Order::getRowMajorID(4)));
    t.populate(data);
    t.setOrder(mv::Order(mv::Order::getColMajorID(4)));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ((double)t(i), reorderedData[i]);

}

TEST(tensor, column_major_to_planar)
{

    mv::Shape tShape({3, 3, 3, 3});
    std::vector<double> data = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f,
        65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f,
        73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)));
    t.populate(data);
    t.setOrder(mv::Order("HWCN"));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_TRUE(t(i) == reorderedData[i]);

}

TEST(tensor, planar_to_column_major)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f,
        65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f,
        73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order("HWCN"));
    t.populate(data);
    t.setOrder(mv::Order(mv::Order::getColMajorID(4)));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_TRUE(t(i) == reorderedData[i]);

}

TEST(tensor, row_major_to_planar)
{

    mv::Shape tShape({3, 3, 3, 3});
    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order(mv::Order::getRowMajorID(4)));
    t.populate(data);
    t.setOrder(mv::Order("HWCN"));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_TRUE(t(i) == reorderedData[i]);

}

TEST(tensor, planar_to_row_major)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order("HWCN"));
    t.populate(data);
    t.setOrder(mv::Order(mv::Order::getRowMajorID(4)));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_TRUE(t(i) == reorderedData[i]);

}

/*TEST(tensor, ind_to_sub_1d)
{

    mv::Shape tShape({32});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor tColumnMajor("t", tShape, mv::DType("Float16"), mv::Order("CHW"));
    mv::Tensor tRowMajor("t", tShape, mv::DType("Float16"), mv::Order("WHC"));
    mv::Tensor tPlanar("t", tShape, mv::DType("Float16"), mv::Order("HWC"));
    tColumnMajor.populate(data);
    tRowMajor.populate(data);
    tPlanar.populate(data);

    for (unsigned i = 0; i < data.size(); ++i)
    {
        auto subColumnMajor = tColumnMajor.indToSub(i);
        auto subRowMajor = tRowMajor.indToSub(i);
        auto subPlanar = tPlanar.indToSub(i);
        ASSERT_EQ(tColumnMajor(subColumnMajor), data[i]);
        ASSERT_EQ(tRowMajor(subRowMajor), data[i]);
        ASSERT_EQ(tPlanar(subPlanar), data[i]);
    }

}

TEST(tensor, ind_to_sub_2d)
{

    mv::Shape tShape({8, 4});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor tColumnMajor("t", tShape, mv::DType("Float16"), mv::Order("CHW"));
    mv::Tensor tRowMajor("t", tShape, mv::DType("Float16"), mv::Order(Order::getRowMajorID(3)));
    mv::Tensor tPlanar("t", tShape, mv::DType("Float16"), mv::Order("HWC"));
    tColumnMajor.populate(data);
    tRowMajor.populate(data);
    tPlanar.populate(data);

    for (unsigned i = 0; i < data.size(); ++i)
    {
        auto subColumnMajor = tColumnMajor.indToSub(i);
        auto subRowMajor = tRowMajor.indToSub(i);
        auto subPlanar = tPlanar.indToSub(i);

        ASSERT_EQ(tColumnMajor(subColumnMajor), data[i]);
        ASSERT_EQ(tRowMajor(subRowMajor), data[i]);
        ASSERT_EQ(tPlanar(subPlanar), data[i]);
    }

}*/

TEST(tensor, augment)
{

    mv::Shape tShape({8, 1, 4});
    mv::Shape tShapeAugmented({8, 4, 4});

    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order("CHW"));
    t.populate(data);
    t.broadcast(tShapeAugmented);

    for (unsigned k = 0; k < 4; ++k)
        for (unsigned j = 0; j < 4; ++j)
            for (unsigned i = 0; i < 8; ++i)
                ASSERT_TRUE((double)t({i, j, k}) == (double)t({i, 0, k}));

}

TEST(tensor, add)
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape tShape({32, 32, 128});
    std::vector<double> data1 = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);
    std::vector<double> data2 = mv::utils::generateSequence<double>(tShape.totalSize(), -start, -diff);

    mv::Tensor t1("t1", tShape, mv::DType("Float16"), mv::Order("CHW"), data1);
    mv::Tensor t2("t2", tShape, mv::DType("Float16"), mv::Order("CHW"), data2);

    auto t3 = mv::math::add(t1, t2);

    std::cout << t3.getShape().totalSize() << std::endl;

    /*for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ(t3({i, j, k}), 0.0f);*/

}

TEST(tensor, add_broadcast_vec)
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape t1Shape({32, 32, 128});
    mv::Shape t2Shape({128});
    std::vector<double> data1 = mv::utils::generateSequence<double>(t1Shape.totalSize(), start, diff);
    std::vector<double> data2 = mv::utils::generateSequence<double>(t2Shape.totalSize());

    mv::Tensor t1("t1", t1Shape, mv::DType("Float16"), mv::Order("WHC"), data1);
    mv::Tensor t2("t2", t2Shape, mv::DType("Float16"), mv::Order(mv::Order::getRowMajorID(1)), data2);

    auto t3 = mv::math::add(t1, t2);

    std::cout << t3.getShape().totalSize() << std::endl;

    /*for (unsigned i = 0; i < t1Shape[0]; ++i)
        for (unsigned j = 0; j < t1Shape[1]; ++j)
            for (unsigned k = 0; k < t1Shape[2]; ++k)
                ASSERT_FLOAT_EQ(t3({i, j, k}), t1({i, j, k}) + t2(k));*/


}

TEST(tensor, add_broadcast_mat)
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape t1Shape({8, 4, 3});
    mv::Shape t2Shape({4, 3});
    std::vector<double> data1 = mv::utils::generateSequence<double>(t1Shape.totalSize(), start, diff);
    std::vector<double> data2 = mv::utils::generateSequence<double>(t2Shape.totalSize());

    mv::Tensor t1("t1", t1Shape, mv::DType("Float16"), mv::Order("CHW"), data1);
    mv::Tensor t2("t2", t2Shape, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(2)), data2);

    auto t3 = mv::math::add(t1, t2);

    for (unsigned i = 0; i < t1Shape[0]; ++i)
        for (unsigned j = 0; j < t2Shape[0]; ++j)
            for (unsigned k = 0; k < t1Shape[2]; ++k)
            {
                float res = (float) t1({i, j, k}) + (float)t2({j, k});
                ASSERT_FLOAT_EQ( (float) t3({i, j, k}), res);
            }

}

/*TEST(tensor, add_broadcast_eq)
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape t1Shape({32, 3, 3, 16});
    mv::Shape t2Shape({32, 3, 3, 1});
    std::vector<double> data1 = mv::utils::generateSequence<double>(t1Shape.totalSize(), start, diff);
    std::vector<double> data2 = mv::utils::generateSequence<double>(t2Shape.totalSize());

    mv::Tensor t1("t1", t1Shape, mv::DType("Float16"), mv::Order("CHW"), data1);
    mv::Tensor t2("t2", t2Shape, mv::DType("Float16"), mv::Order("CHW"), data2);

    auto t3 = mv::math::add(t1, t2);

    for (unsigned i = 0; i < t1Shape[0]; ++i)
        for (unsigned j = 0; j < t2Shape[1]; ++j)
            for (unsigned k = 0; k < t1Shape[2]; ++k)
                for (unsigned l = 0; l < t2Shape[3]; ++l)
                    ASSERT_FLOAT_EQ(t3({i, j, k, l}), t1({i, j, k, l}) + t2({i, j, k, 0}));

}*/

TEST(tensor, subtract)
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape tShape({32, 32, 3});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);

    mv::Tensor t1("t1", tShape, mv::DType("Float16"), mv::Order("CHW"), data);
    mv::Tensor t2("t2", tShape, mv::DType("Float16"), mv::Order("CHW"), data);

    t1.subtract(t2);

    for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ( (float) t1({i, j, k}), 0.0f);

}

TEST(tensor, multiply)
{

    double start = 1.0f;
    double diff = 0.5f;

    mv::Shape tShape({32, 32, 3});
    std::vector<double> data1 = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);
    std::vector<double> data2(data1.size());

    for (unsigned i = 0; i < data2.size(); ++i)
        data2[i] = 1.0f / data1[i];

    mv::Tensor t1("t1", tShape, mv::DType("Float16"), mv::Order("CHW"), data1);
    mv::Tensor t2("t2", tShape, mv::DType("Float16"), mv::Order("CHW"), data2);

    t1.multiply(t2);

    for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ( (float) t1({i, j, k}), 1.0f);

}

TEST(tensor, divide)
{

    double start = 2.0f;
    double diff = 0.5f;

    mv::Shape tShape({32, 32, 3});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);

    mv::Tensor t1("t1", tShape, mv::DType("Float16"), mv::Order("CHW"), data);
    mv::Tensor t2("t2", tShape, mv::DType("Float16"), mv::Order("CHW"), data);

    t1.divide(t2);

    for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ( (float) t1({i, j, k}), 1.0f);

}

TEST(tensor, get_data)
{

    double start = 2.0f;
    double diff = 0.5f;

    mv::Shape tShape({32, 32, 128});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);

    mv::Tensor t1("t1", tShape, mv::DType("Float16"), mv::Order("CHW"), data);

    std::cout << t1.getDoubleData().size() << std::endl;

}

TEST(tensor, to_binary_fp16)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DType("Float16"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    mv_num_convert cvtr;
    const std::vector<int16_t>& fp16_data = bdata.fp16();
    for (unsigned i = 0; i < fp16_data.size(); i++)
        EXPECT_TRUE(fp16_data[i] == cvtr.fp32_to_fp16(data[i]));
}

TEST(tensor, to_binary_fp64)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DType("Float64"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<double>& fp64_data = bdata.fp64();
    for (unsigned i = 0; i < fp64_data.size(); i++)
        EXPECT_TRUE(fp64_data[i] ==  data[i]);
}

TEST(tensor, to_binary_fp32)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DType("Float32"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<float>& fp32_data = bdata.fp32();
    for (unsigned i = 0; i < fp32_data.size(); i++)
        EXPECT_TRUE(fp32_data[i] == data[i]);
}

TEST(tensor, to_binary_u64)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<int64_t> data = {
        0, 27, 54, 9, 36, 63, 18, 45, 72,
        1, 28, 55, 10, 37, 64, 19, 46, 73,
        2, 29, 56, 11, 38, 65, 20, 47, 74,
        3, 30, 57, 12, 39, 66, 21, 48, 75,
        4, 31, 58, 13, 40, 67, 22, 49, 76,
        5, 32, 59, 14, 41, 68, 23, 50, 77,
        6, 33, 60, 15, 42, 69, 24, 51, 78,
        7, 34, 61, 16, 43, 70, 25, 52, 79,
        8, 35, 62, 17, 44, 71, 26, 53, 80
    };

    mv::Tensor t("t", tShape, mv::DType("UInt64"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<uint64_t>& u64_data = bdata.u64();
    for (unsigned i = 0; i < u64_data.size(); i++)
        EXPECT_EQ(u64_data[i], (uint64_t) data[i]);
}

TEST(tensor, to_binary_u32)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<int64_t> data = {
        0, 27, 54, 9, 36, 63, 18, 45, 72,
        1, 28, 55, 10, 37, 64, 19, 46, 73,
        2, 29, 56, 11, 38, 65, 20, 47, 74,
        3, 30, 57, 12, 39, 66, 21, 48, 75,
        4, 31, 58, 13, 40, 67, 22, 49, 76,
        5, 32, 59, 14, 41, 68, 23, 50, 77,
        6, 33, 60, 15, 42, 69, 24, 51, 78,
        7, 34, 61, 16, 43, 70, 25, 52, 79,
        8, 35, 62, 17, 44, 71, 26, 53, 80
    };

    mv::Tensor t("t", tShape, mv::DType("UInt32"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<uint32_t>& u32_data = bdata.u32();
    for (unsigned i = 0; i < u32_data.size(); i++)
        EXPECT_EQ(u32_data[i], (uint32_t)data[i]);
}

TEST(tensor, to_binary_u16)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<int64_t> data = {
        0, 27, 54, 9, 36, 63, 18, 45, 72,
        1, 28, 55, 10, 37, 64, 19, 46, 73,
        2, 29, 56, 11, 38, 65, 20, 47, 74,
        3, 30, 57, 12, 39, 66, 21, 48, 75,
        4, 31, 58, 13, 40, 67, 22, 49, 76,
        5, 32, 59, 14, 41, 68, 23, 50, 77,
        6, 33, 60, 15, 42, 69, 24, 51, 78,
        7, 34, 61, 16, 43, 70, 25, 52, 79,
        8, 35, 62, 17, 44, 71, 26, 53, 80
    };

    mv::Tensor t("t", tShape, mv::DType("UInt16"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<uint16_t>& u16_data = bdata.u16();
    for (unsigned i = 0; i < u16_data.size(); i++)
        EXPECT_EQ(u16_data[i], (uint16_t) data[i]);
}

TEST(tensor, to_binary_u8)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<int64_t> data = {
        0, 27, 54,  9, 36, 63, 18, 45, 72,
        1, 28, 55, 10, 37, 64, 19, 46, 73,
        2, 29, 56, 11, 38, 65, 20, 47, 74,
        3, 30, 57, 12, 39, 66, 21, 48, 75,
        4, 31, 58, 13, 40, 67, 22, 49, 76,
        5, 32, 59, 14, 41, 68, 23, 50, 77,
        6, 33, 60, 15, 42, 69, 24, 51, 78,
        7, 34, 61, 16, 43, 70, 25, 52, 79,
        8, 35, 62, 17, 44, 71, 26, 53, 80
    };

    mv::Tensor t("t", tShape, mv::DType("UInt8"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<uint8_t>& u8_data = bdata.u8();
    for (unsigned i = 0; i < u8_data.size(); i++)
        EXPECT_EQ(u8_data[i], (uint8_t) data[i]);
}

TEST(tensor, to_binary_i64)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<int64_t> data = {
        0, 27, 54,  9, 36, 63, 18, 45, 72,
        1, 28, 55, 10, 37, 64, 19, 46, 73,
        2, 29, 56, 11, 38, 65, 20, 47, 74,
        3, 30, 57, 12, 39, 66, 21, 48, 75,
        4, 31, 58, 13, 40, 67, 22, 49, 76,
        5, 32, 59, 14, 41, 68, 23, 50, 77,
        6, 33, 60, 15, 42, 69, 24, 51, 78,
        7, 34, 61, 16, 43, 70, 25, 52, 79,
        8, 35, 62, 17, 44, 71, 26, 53, 80
    };

    mv::Tensor t("t", tShape, mv::DType("Int64"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<int64_t>& i64_data = bdata.i64();
    for (unsigned i = 0; i < i64_data.size(); i++)
        EXPECT_EQ(i64_data[i], (int64_t) data[i]);
}

TEST(tensor, to_binary_i32)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<int64_t> data = {
        0, 27, -54, -9, 36, 63, 18, 45, 72,
        1, 28, -55, -10, 37, 64, 19, 46, 73,
        2, 29, -56, -11, 38, 65, 20, 47, 74,
        3, 30, -57, -12, 39, 66, 21, 48, 75,
        4, 31, -58, -13, 40, 67, 22, 49, 76,
        5, 32, -59, -14, 41, 68, 23, 50, 77,
        6, 33, -60, -15, 42, 69, 24, 51, 78,
        7, 34, -61, -16, 43, 70, 25, 52, 79,
        8, 35, -62, -17, 44, 71, 26, 53, 80
    };

    mv::Tensor t("t", tShape, mv::DType("Int32"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<int32_t>& i32_data = bdata.i32();
    for (unsigned i = 0; i < i32_data.size(); i++)
        EXPECT_EQ(i32_data[i], (int32_t) data[i]);
}

TEST(tensor, to_binary_i16)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<int64_t> data = {
        0, 27, -54, -9, 36, 63, 18, 45, 72,
        1, 28, -55, -10, 37, 64, 19, 46, 73,
        2, 29, -56, -11, 38, 65, 20, 47, 74,
        3, 30, -57, -12, 39, 66, 21, 48, 75,
        4, 31, -58, -13, 40, 67, 22, 49, 76,
        5, 32, -59, -14, 41, 68, 23, 50, 77,
        6, 33, -60, -15, 42, 69, 24, 51, 78,
        7, 34, -61, -16, 43, 70, 25, 52, 79,
        8, 35, -62, -17, 44, 71, 26, 53, 80
    };

    mv::Tensor t("t", tShape, mv::DType("Int16"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<int16_t>& i16_data = bdata.i16();
    for (unsigned i = 0; i < i16_data.size(); i++)
        EXPECT_EQ(i16_data[i], (int16_t) data[i]);
}

TEST(tensor, to_binary_i8)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<int64_t> data = {
         0, 27, -54, -9, 36, 63, 18, 45, 72,
        1, 28, -55, -10, 37, 64, 19, 46, 73,
        2, 29, -56, -11, 38, 65, 20, 47, 74,
        3, 30, -57, -12, 39, 66, 21, 48, 75,
        4, 31, -58, -13, 40, 67, 22, 49, 76,
        5, 32, -59, -14, 41, 68, 23, 50, 77,
        6, 33, -60, -15, 42, 69, 24, 51, 78,
        7, 34, -61, -16, 43, 70, 25, 52, 79,
        8, 35, -62, -17, 44, 71, 26, 53, 80
    };

    mv::Tensor t("t", tShape, mv::DType("Int8"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<int8_t>& i8_data = bdata.i8();
    for (unsigned i = 0; i < i8_data.size(); i++)
        EXPECT_EQ(i8_data[i], (int8_t) data[i]);
}

TEST(tensor, to_binary_i4)
{

    mv::Shape tShape({1, 1, 1, 16});

    std::vector<int64_t> data = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15
    };

    mv::Tensor t("t", tShape, mv::DType("Int4"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<int8_t>& i4_data = bdata.i4();
    EXPECT_EQ(i4_data.size(), 8);
    int idx = 0;
    for (unsigned i = 0; i < i4_data.size(); i++)
    {
        EXPECT_EQ((i4_data[i] & 0xF), (int8_t) data[idx++]);
        EXPECT_EQ((i4_data[i] & 0xF0) >> 4, (int8_t) data[idx++]);
    }
}

TEST(tensor, to_binary_i2)
{

    mv::Shape tShape({1, 1, 1, 16});

    std::vector<int64_t> data = {
        0,
        1,
        2,
        3,
        0,
        1,
        1,
        3,
        0,
        2,
        2,
        1,
        3,
        2,
        1,
        0
    };

    mv::Tensor t("t", tShape, mv::DType("Int2"), mv::Order("HWCN"));
    t.populate(data);
    mv::BinaryData bdata = t.toBinary();
    const std::vector<int8_t>& i2_data = bdata.i2();
    EXPECT_EQ(i2_data.size(), 4);
    int idx = 0;
    for (unsigned i = 0; i < i2_data.size(); i++)
    {
        EXPECT_EQ((i2_data[i] & 0x3), (int8_t) data[idx++]);
        EXPECT_EQ((i2_data[i] & 0xC) >> 2, (int8_t) data[idx++]);
        EXPECT_EQ((i2_data[i] & 0x30) >> 4, (int8_t) data[idx++]);
        EXPECT_EQ((i2_data[i] & 0xC0) >> 6, (int8_t) data[idx++]);
    }
}

TEST(tensor, to_binary_i2x_invalid_execution)
{

    mv::Shape tShape({1, 1, 1, 16});

    std::vector<int64_t> data = {
        0,
        1,
        2,
        3,
        0,
        1,
        1,
        3,
        0,
        2,
        2,
        1,
        3,
        2,
        1,
        0
    };

    mv::Tensor t("t", tShape, mv::DType("Int2X"), mv::Order("HWCN"));
    t.populate(data);
    ASSERT_ANY_THROW(t.toBinary());
}

TEST(tensor, sparsity)
{
    //Example of weights in res2a_branch2a
    mv::Shape tShape({1, 1, 64, 64});
    mv::Tensor t("res2a_branch2a_weigths", tShape, mv::DType("UInt8"), mv::Order("NHWC"));
    std::ifstream inputfile(mv::utils::projectRootPath() + std::string("/tests/data/res2a_branch2a_weigths_input.bin"), std::ios::binary );

    uint8_t a;
    size_t count = 0;
    std::vector<int64_t> indata(t.getShape().totalSize());
    while(inputfile.read(reinterpret_cast<char *>(&a), sizeof(a)))
    {
        indata[count++] = a;
    }
    t.populate(indata);

    mv::QuantizationParams q({122}, {0.00282943}, {0},{0});
    t.set<mv::QuantizationParams>("quantizationParams",q);
    ASSERT_NO_THROW(t.setSparse());
    ASSERT_TRUE(t.isSparse());
    std::shared_ptr<mv::Tensor> sparsityMap = t.getSparsityMap();

    std::vector<int64_t> res = sparsityMap->getIntData();
    std::vector<int64_t> data_res = t.getIntData();
    //reference result of sparsity map
    std::ifstream outputfile(mv::utils::projectRootPath() + std::string("/tests/data/res2a_branch2a_weigths_output.bin"), std::ios::binary );

    std::vector<double> refdata(res.size());
    count = 0;
    while(outputfile.read(reinterpret_cast<char *>(&a), sizeof(a)))
    {
        refdata[count++] = a;
    }

    ASSERT_TRUE(count == res.size());
    //channelTracker is a counter used to scan through the channels
    uint64_t channelTracker = 0;
    //counter is used to track non-padding values
    uint64_t counter = 0;
    for (unsigned i = 0; i < res.size(); ++i)
    {
        ASSERT_EQ(res[i], refdata[i]);
        for (size_t k=0; k < 8; k++)
        {
            if (channelTracker == t.getShape()[(t.getShape().ndims())-1])
            {
                channelTracker = 0;
                i = i + i%16;
            }
            if (i < res.size())
            {
                channelTracker++;
                if (data_res[counter*8+k] == 122)
                    ASSERT_TRUE((static_cast<uint8_t>(res[i]) & (1<<k)) == 0);
                if (data_res[counter*8+k] != 122)
                    ASSERT_TRUE((static_cast<uint8_t>(res[i]) & (1<<k)) != 0);
            }
        }
        counter++;
    }
    
    mv::Shape seShape({1,1,1,64});
    ASSERT_TRUE(seShape == t.getStorageElement()->getShape());

    mv::Shape sparsityMapShape({16,1,1,64});
    ASSERT_TRUE(sparsityMapShape == sparsityMap->getShape());
}

TEST(tensor, sparsity_res3a_branch2c)
{
    //Example of weights in res3a_branch2c
    mv::Shape tShape({1, 1, 128, 512});
    mv::Tensor t("res3a_branch2c_weigths", tShape, mv::DType("UInt8"), mv::Order("NHWC"));
    std::ifstream inputfile(mv::utils::projectRootPath() + std::string("/tests/data/res3a_branch2c_weigths_input.bin"), std::ios::binary );

    uint8_t a;
    size_t count = 0;
    std::vector<int64_t> indata(t.getShape().totalSize());
    while(inputfile.read(reinterpret_cast<char *>(&a), sizeof(a)))
    {
        indata[count++] = a;
    }
    t.populate(indata);

    mv::QuantizationParams q({137}, {0.00361593}, {0},{0});
    t.set<mv::QuantizationParams>("quantizationParams",q);
    ASSERT_NO_THROW(t.setSparse());
    ASSERT_TRUE(t.isSparse());
    std::shared_ptr<mv::Tensor> sparsityMap = t.getSparsityMap();

    std::vector<int64_t> res = sparsityMap->getIntData();
    std::vector<int64_t> data_res = t.getIntData();

    //reference result of sparsity map
    std::ifstream outputfile(mv::utils::projectRootPath() + std::string("/tests/data/res3a_branch2c_weigths_output.bin"), std::ios::binary );

    std::vector<double> refdata(res.size());
    count = 0;
    while(outputfile.read(reinterpret_cast<char *>(&a), sizeof(a)))
    {
        refdata[count++] = a;
    }

    ASSERT_TRUE(count == res.size());
    for (unsigned i = 0; i < res.size(); ++i)
    {
        for (size_t k=0; k < 8; k++)
        {
            if (data_res[i*8+k] == 137)
                ASSERT_TRUE((static_cast<uint8_t>(res[i]) & (1<<k)) == 0);
            if (data_res[i*8+k] != 137)
                ASSERT_TRUE((static_cast<uint8_t>(res[i]) & (1<<k)) != 0);
        }
        ASSERT_EQ(res[i], refdata[i]);
    }

    mv::Shape seShape({1,1,1,512});
    ASSERT_TRUE(seShape == t.getStorageElement()->getShape());

    mv::Shape sparsityMapShape({1,1,16,512});
    ASSERT_TRUE(sparsityMapShape == sparsityMap->getShape());

    std::vector<mv::DataElement> denseData = t.getDataPacked();
    count = 0;
    size_t j = 0;
    size_t padsize = 0;
    for (unsigned i = 0; i < data_res.size(); ++i)
    {
        while (denseData[j] == 137 && j < denseData.size()) //its padding ignore it
        {
            j++;
            padsize++;
        }
        if (data_res[i] != 137)
        {
            ASSERT_TRUE(data_res[i] == (int64_t) denseData[j]);
            j++;
        }
        else
        {
            count++;
        }

    }
    ASSERT_TRUE((count + denseData.size() - padsize) == data_res.size());
}

//VPUNND-391
TEST(tensor, testing_at)
{
    mv::Shape tShape({1, 1, 128, 512});
    mv::Tensor t("test", tShape, mv::DType("Float8"), mv::Order("WCNH"));
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    t.populate(data);
    std::vector<double> resdata = t.getDoubleData();
    for(size_t i=0; i< resdata.size();i++)
        ASSERT_TRUE(t.at(i) == resdata[i]);

    double val = t.at(5);
    t.at(5) = val * 5;
    ASSERT_TRUE(t.at(5) == val*5);

    t(10) = (int64_t)20;
    ASSERT_TRUE(t.at(10) == 20);

    val = t(20);
    t(10) *= 20.3;
    ASSERT_TRUE(t.at(10) == val*20.3);

    mv::Tensor ti("test", tShape, mv::DType("Int32"), mv::Order("WCNH"));
    std::vector<int64_t> datai = mv::utils::generateSequence<int64_t>(tShape.totalSize());
    ti.populate(datai);

    ti(10) = 20.2;
    ASSERT_TRUE(ti.at(10) == 20);

    val = t(20);
    ti(20) *= 20.3;
    ASSERT_TRUE(ti.at(20) == (int64_t)(val*20.3));
}
