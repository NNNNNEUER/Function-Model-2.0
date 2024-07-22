#include "CU.hpp"
#include "DLU.hpp"
#include "LDTU.hpp"
#include "RDTU.hpp"
#include "VU.hpp"
#include <iostream>
#include <ctime>
#include <cassert>
#include <format>

L2 l2_i(1024 * 1024);                   // unit: 1 byte
DLU dlu_i(l2_i, 64 * 1024);             // unit: 16 bytes
LDTU ldtu_i(dlu_i, 4 * 1024);           // unit: 16 bytes
RDTU rdtu_i(dlu_i, 4 * 1024, 2 * 1024); // unit: 16 bytes
CU cu_i(ldtu_i, rdtu_i, 8 * 1024);      // unit: 16 bytes
VU vu_i(cu_i, dlu_i, l2_i);

#define CEIL(X, Y) ((X + Y - 1) / Y)
#define FEATURE_INDEX(i, j, k) (i * W * C + j * C + k)
#define WEIGHT_INDEX(oi, ki, kj, ii) \
    (oi * Kh * Kw * Ci + ki * Kw * Ci + kj * Ci + ii)

// How to partion the feature and weight
size_t Co0 = 5;
size_t H0 = 8;
size_t W0 = 3;
size_t Ci0 = C0;

// ----- Feature in L2 -----
size_t stride = 1;
size_t H = 20;
size_t W = 18;
size_t C = 18;
size_t feature_l2_addr = 0;
size_t feature_l1_addr = 0;

// ----- Weight in L2 -----
size_t Co = 19;
size_t Kh = 3;
size_t Kw = Kh;
size_t Ci = C;
size_t weight_l2_addr = H * W * C;
size_t weight_l1_addr = (H0 + Kh - 1) * (W0 + Kw - 1);

// ----- Bias in L2 -----
size_t Co1 = CEIL(Co, Co0);
size_t H1 = CEIL(H, H0);
size_t W1 = CEIL(W, W0);
size_t Ci1 = CEIL(Ci, C0);
size_t bias_l2_addr = weight_l2_addr + Co * Kh * Kw * Ci;
size_t bias_l1_addr = weight_l1_addr + Co0 * Kh * Kw;

// ----- partition -----
size_t M = H0 * W0;
size_t K = Kh * Kw * Ci0;
size_t N = Co0;
size_t M0 = C0;
size_t N0 = C0;
size_t K0 = Ci0;

// ----- computation -----
size_t M1 = CEIL(M, M0);
size_t N1 = CEIL(N, N0);
size_t K1 = CEIL(K, K0);

std::vector<uint8_t> golden_WconvF(std::vector<uint8_t> &f, std::vector<uint8_t> &w, std::vector<uint8_t> &b)
{
    // kernel should be a square matrix whose sidelength should be an odd number
    size_t retH = (H - Kh + Kh / 2 * 2) / stride + 1;
    size_t retW = (W - Kw + Kw / 2 * 2) / stride + 1;
    std::vector<uint8_t> res(retH * retW * Co);
    auto row_pad = Kh / 2;
    auto col_pad = Kw / 2;
    for (size_t co = 0; co < Co; ++co)
    {
        for (size_t i = 0; i < retH; ++i)
        {
            for (size_t j = 0; j < retW; ++j)
            {
                res.at(i * retW * Co + j * Co + co) = b.at(co);
                for (size_t ci = 0; ci < Ci; ++ci)
                {
                    for (size_t ki = 0; ki < Kh; ++ki)
                    {
                        for (size_t kj = 0; kj < Kw; ++kj)
                        {
                            size_t im_row = stride * i - row_pad + ki;
                            size_t im_col = stride * j - col_pad + kj;
                            if (im_row < H && im_col < W && ci < C)
                            {
                                res.at(i * retW * Co + j * Co + co) += f.at(im_row * W * C + im_col * C + ci) * w.at(co * Kh * Kw * Ci + ki * Kw * Ci + kj * Ci + ci);
                            }
                            else
                            {
                                res.at(i * retW * Co + j * Co + co) += 0;
                            }
                        }
                    }
                }
            }
        }
    }
    return res;
}

void display(std::vector<uint8_t> &f, size_t c)
{
    size_t retH = (H - Kh + Kh / 2 * 2) / stride + 1;
    size_t retW = (W - Kw + Kw / 2 * 2) / stride + 1;
    assert(c < Co);
    std::cerr << std::format("H: {}, W: {}", retH, retW) << "\n";
    std::cerr << std::format("Channel: {}", c) << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < retW; ++i)
    {
        std::cerr << std::format("{:5}", i);
    }
    std::cerr << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < retW; ++i)
    {
        std::cerr << "  ---";
    }
    std::cerr << "\n";
    for (size_t i = 0; i < retH; ++i)
    {
        std::cerr << std::format("{:5}|", i);
        for (size_t j = 0; j < retW; ++j)
        {
            std::cerr << std::format("{:5}", f.at(i * retW * Co + j * Co + c));
        }
        std::cerr << "\n";
    }
    std::cerr << std::endl;
}

void checkDisplay(std::vector<uint8_t> &golden_res, std::vector<uint8_t> &res)
{
    size_t retW = (W - Kw + Kw / 2 * 2) / stride + 1;
    for (size_t k = 0; k < Co; ++k)
    {
        std::cerr << "reference: " << std::endl;
        display(golden_res, k);
        std::cerr << "result: " << std::endl;
        display(res, k);
        for (size_t i = 0; i < H; ++i)
            for (size_t j = 0; j < W; ++j)
                assert(golden_res.at(i * retW * Co + j * Co + k) == res.at(i * retW * Co + j * Co + k));
    }
}

int main()
{
    srand(time(NULL));

    // ----- Feature in L2 -----
    std::vector<uint8_t> feature(H * W * C);
    for (size_t i = 0; i < H; ++i)
    {
        for (size_t j = 0; j < W; ++j)
            for (size_t k = 0; k < C; ++k)
            {
                auto val = rand() % 256;
                l2_i.write(feature_l2_addr + FEATURE_INDEX(i, j, k), val);
                feature.at(i * W * C + j * C + k) = val;
            }
    }

    // ----- Weight in L2 -----
    std::vector<uint8_t> weight(Co * Kh * Kw * Ci);
    for (size_t oi = 0; oi < Co; ++oi)
    {
        for (size_t ki = 0; ki < Kh; ++ki)
            for (size_t kj = 0; kj < Kw; ++kj)
                for (size_t ii = 0; ii < Ci; ++ii)
                {
                    auto val = rand() % 256;
                    l2_i.write(weight_l2_addr + WEIGHT_INDEX(oi, ki, kj, ii), val);
                    weight.at(oi * Kh * Kw * Ci + ki * Kw * Ci + kj * Ci + ii) = val;
                }
    }

    // ----- Bias in L2 -----
    std::vector<uint8_t> bias(Co);
    for (size_t i = 0; i < Co; ++i)
    {
        auto val = rand() % 256;
        l2_i.write(bias_l2_addr + i, val);
        bias.at(i) = val;
    }

    // For easy comparison, use Feature to store PSB transferred results
    std::vector<uint8_t> res(H * W * Co);

    for (size_t oi = 0; oi < Co1; ++oi)
        for (size_t i = 0; i < H1; ++i)
            for (size_t j = 0; j < W1; ++j)
            {
                // clear PSB,ii loop is finished and PSB is the result of a (H0,W0,Co0) sub-block
                // if Co0 < C0 (N < N0) then there will be padding
                dlu_i.loadSubBias(bias_l1_addr, oi, bias_l2_addr, Co, Co0);
                rdtu_i.loadMMB(bias_l1_addr, oi);
                cu_i.presetBias(oi, M, N);

                // Continue to do Ci1 times to complete the accumulation of Ci dimensions
                for (size_t ii = 0; ii < Ci1; ++ii)
                {
                    dlu_i.loadSubPaddedFeature(feature_l1_addr, i, j, ii, feature_l2_addr, H, W, C, H0, W0, Kh, Kw, C0);
                    dlu_i.loadSubWeight(weight_l1_addr, oi, ii, weight_l2_addr, Co, Ci, Co0, Kh, Kw, Ci0);
                    ldtu_i.load_im2col(feature_l1_addr, H0, W0, Kh, Kw);
                    rdtu_i.load(weight_l1_addr, Co0, Kh, Kw);

                    // CU calculation
                    // Done computation along the H0*W0 dimension
                    for (size_t i = 0; i < M1; ++i)
                        // finished adding Kh*Kw dimensions
                        for (size_t j = 0; j < K1; ++j)
                            // Done computation along the Co0 dimension
                            for (size_t k = 0; k < N1; ++k)
                                cu_i.matmul(i, j, k, M0, K0, N0, M, K, N);
                }
                vu_i.LoadOut(res, i, j, oi, H, W, Co, H0, W0, C0, Co0);
            }

    // ----- check -----
    auto ref_result = golden_WconvF(feature, weight, bias);
    checkDisplay(ref_result, res);
    puts("Passed");
}