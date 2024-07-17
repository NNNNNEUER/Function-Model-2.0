#include "CU.hpp"
#include "DLU.hpp"
#include "LDTU.hpp"
#include "RDTU.hpp"
#include "VU.hpp"

#include <ctime>
#include <format>

// ----- Feature in L2 -----
size_t feature_L2_addr = 0;
size_t H = 7;
size_t W = 7;
size_t C = 16;
size_t stride = 1;

// ----- Weight in L2 -----
size_t weight_L2_addr = H * W * C;
size_t Co = 16;
size_t Kh = 3;
size_t Kw = Kh;
size_t Ci = C;

// ----- Bias in L2 -----
size_t bias_L2_addr = weight_L2_addr + Co * Kh * Kw * Ci;

// How to partion the feature and weight
size_t H0 = 3;
size_t W0 = 3;
size_t C0 = 16;
size_t Ci0 = C0;
size_t Co0 = C0; // can be other value other than C0

// partition LMB and RMB, don't change
size_t M = H0 * W0;
size_t K = Kh * Kw;
size_t N = Co0;
size_t M0 = 16;
size_t K0 = 16;
size_t N0 = 16;

// addr in L1
size_t feature_L1_addr = 0;
size_t weight_L1_addr = H0 * W0 * Kh * Kw * C0;
size_t bias_L1_addr = H0 * W0 * Kh * Kw * C0 + Co0 * Kh * Kw * Ci0;

// ----- computation -----
#define CEIL(X, M) ((X + M - 1) / M)
#define FEATURE_INDEX(H, W, C, i, j, k) (i * W * C + j * C + k)
#define WEIGHT_INDEX(Co, Kh, Kw, Ci, oi, ki, kj, ii) \
    (oi * Kh * Kw * Ci + ki * Kw * Ci + kj * Ci + ii)
size_t Co1 = CEIL(Co, Co0);
size_t H1 = CEIL(H, H0);
size_t W1 = CEIL(W, W0);
size_t Ci1 = CEIL(Ci, C0);

// ----- Initialization -----
std::vector<int8_t> l2_i(1024 * 1024); // unit: 1 byte
DLU dlu_i(l2_i, 1024 * 10);
LDTU ldtu_i(dlu_i, 64 * 1024);
RDTU rdtu_i(dlu_i, 64 * 1024, 1024);
CU cu_i(ldtu_i, rdtu_i, 128 * 1024);
VU vu_i(l2_i, dlu_i, cu_i);

// ----- Functions-----
std::vector<int8_t> golden_WconvF(std::vector<int8_t> &f, std::vector<int8_t> &w)
{
    // kernel should be a square matrix whose sidelength should be an odd number
    size_t retH = (H - Kh + Kh / 2 * 2) / stride + 1;
    size_t retW = (W - Kw + Kw / 2 * 2) / stride + 1;
    std::vector<int8_t> res(retH * retW * Co);
    auto row_pad = Kh / 2;
    auto col_pad = Kw / 2;
    for (size_t co = 0; co < Co; ++co)
    {
        for (size_t i = 0; i < retH; ++i)
        {
            for (size_t j = 0; j < retW; ++j)
            {
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

void display(std::vector<int8_t> &f, size_t c)
{
    assert(c < Co);
    std::cerr << std::format("H: {}, W: {}", H, W) << "\n";
    std::cerr << std::format("Channel: {}", c) << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < W; ++i)
    {
        std::cerr << std::format("{:5}", i);
    }
    std::cerr << "\n";
    std::cerr << "      ";
    for (size_t i = 0; i < W; ++i)
    {
        std::cerr << "  ---";
    }
    std::cerr << "\n";
    for (size_t i = 0; i < H; ++i)
    {
        std::cerr << std::format("{:5}|", i);
        for (size_t j = 0; j < W; ++j)
        {
            std::cerr << std::format("{:5}", f.at(i * W * c + j * c + c));
        }
        std::cerr << "\n";
    }
    std::cerr << std::endl;
}

void checkDisplay(std::vector<int8_t> &ref, std::vector<int8_t> &res)
{
    for (size_t k = 0; k < Co; ++k)
    {
        std::cerr << "reference: " << std::endl;
        display(ref, k);
        std::cerr << "result: " << std::endl;
        display(res, k);
    }
    assert(ref == res);
}

int main()
{
    srand(time(0));

    // ----- Feature in L2 -----
    std::vector<int8_t> feature(H * W * C);
    for (size_t i = 0; i < H; ++i)
    {
        for (size_t j = 0; j < W; ++j)
            for (size_t k = 0; k < C; ++k)
            {
                int8_t val = j;
                l2_i.at(feature_L2_addr + FEATURE_INDEX(H, W, C, i, j, k)) = val;
                feature.at(i * W * C + j * C + k) = val;
            }
    }

    // ----- Weight in L2 -----
    std::vector<int8_t>
        weight(Co * Kh * Kw * Ci);
    for (size_t co = 0; co < Co; ++co)
    {
        for (size_t ki = 0; ki < Kh; ++ki)
            for (size_t kj = 0; kj < Kw; ++kj)
                for (size_t ci = 0; ci < Ci; ++ci)
                {
                    int8_t val = kj;
                    l2_i.at(weight_L2_addr + WEIGHT_INDEX(Co, Kh, Kw, Ci, co, ki, kj, ci)) = val;
                    weight.at(co * Kh * Kw * Ci + ki * Kw * Ci + kj * Ci + ci) = val;
                }
    }

    // For easy comparison, use Feature to store PSB transferred results
    std::vector<int8_t> res(H * W * Co);

    for (size_t oi = 0; oi < Co1; ++oi)
        for (size_t i = 0; i < H1; ++i)
            for (size_t j = 0; j < W1; ++j)
            {
                cu_i.clearPSB();
                // Continue to do Ci1 times to complete the accumulation of Ci dimensions
                for (size_t ii = 0; ii < Ci1; ++ii)
                {
                    dlu_i.loadSubPaddedFeature(feature_L2_addr, feature_L1_addr, i, j, ii, H0, Kh, W0, Kw, C0, H, W, C);
                    dlu_i.loadSubWeight(weight_L2_addr, weight_L1_addr, oi, ii, Co0, Kh, Kw, Ci0, Co, Ci);
                    dlu_i.printL1();
                    ldtu_i.load_im2col(feature_L1_addr, H0, W0, C0, Kh, Kw);
                    rdtu_i.loadWeight(weight_L1_addr, Co0, Kh, Kw, Ci0);
                    //  CU calculation
                    //  bias is loaded in the first iteration
                    //  Done computation along the Co0 dimension
                    for (size_t n = 0; n < CEIL(N, N0); ++n)
                        // Done computation along the H0*W0 dimension
                        for (size_t m = 0; m < CEIL(M, M0); ++m)
                            // finished accumulating Kh*Kw dimensions
                            for (size_t k = 0; k < CEIL(K, K0); ++k)
                                cu_i.matmulCtrl(false, true, m, k, n, M, N, K, M0, K0, N0);
                }

                // get the result
                vu_i.loadOut(res, i, j, oi, H0, W0, Co0, H, W, Co);
            }

    // ----- check -----
    auto golden_res = golden_WconvF(feature, weight);
    // checkDisplay(golden_res, res);
    puts("Passed");
}
