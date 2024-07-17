#ifndef VU_HPP
#define VU_HPP

#include "DLU.hpp"
#include "CU.hpp"

class VU
{
public:
    VU(std::vector<int8_t> &L2_cache, DLU &L1_cache, CU &psb)
        : L2_cache(L2_cache), L1_cache(L1_cache), PSB_cache(psb) {}

    // PSB transfer (H1, W1, Co1)th (M, N) = (H0*W0, Co0) to Feature(H, W, Co)
    void loadOut(std::vector<int8_t> &res, size_t hi, size_t wj, size_t cok,
                 size_t H0, size_t W0, size_t Co0,
                 size_t H, size_t W, size_t Co)
    {
        // M = H0 * W0
        // N = Co0
        // PSB: M * N
        // Feature: H0 * W0 * Co0
        for (size_t i = 0; i < H0; ++i)
            for (size_t j = 0; j < W0; ++j)
                for (size_t k = 0; k < Co0; ++k)
                {
                    auto H_i = H0 * hi + i;
                    auto W_j = W0 * wj + j;
                    auto C_k = Co0 * cok + k;
                    if (H_i < H && W_j < W && C_k < Co)
                        res.at(H_i * W * Co + W_j * Co + C_k) =
                            PSB_cache.getPSB()[(i * W0 + j) * Co0 + k];
                }
    }

    // TODO: implement the following functions
    void loadtoL2() {}
    void loadtoL1() {}

private:
    std::vector<int8_t> &L2_cache;
    DLU &L1_cache;
    CU &PSB_cache;
};

#endif // VU_HPP