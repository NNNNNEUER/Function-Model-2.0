#ifndef VU_HPP
#define VU_HPP

#include "CU.hpp"
#include "DLU.hpp"
#include "L2.hpp"

#define CEIL(X, Y) ((X + Y - 1) / Y)

class VU
{
public:
    VU(CU &CU, DLU &DLU, L2 &L2) : PSB(CU), L1(DLU), L2(L2) {}

    // PSB transfer (H0,W0,Co0) to std::vector<uint8_t> res
    void LoadOut(std::vector<uint8_t> &res, size_t i, size_t j, size_t oi,
                 size_t H, size_t W, size_t Co, size_t H0, size_t W0, size_t C0, size_t Co0)
    {
        for (size_t sub_i = 0; sub_i < H0; ++sub_i)
            for (size_t sub_j = 0; sub_j < W0; ++sub_j)
                for (size_t co_i = 0; co_i < CEIL(Co0, C0); ++co_i)
                    for (size_t sub_k = 0; sub_k < C0; ++sub_k)
                    {
                        auto im_row = H0 * i + sub_i;
                        auto im_col = W0 * j + sub_j;
                        auto im_ch = (Co0 * oi) + co_i * C0 + sub_k;
                        if (im_row < H && im_col < W && im_ch < Co)
                            // See CU::matmul for details on how (H0,W0,Co0) is stored in PSB
                            res.at(im_row * W * Co + im_col * Co + im_ch) =
                                PSB.getPSB()[(sub_i * W0 + sub_j) * CEIL(Co0, C0) + co_i][sub_k];
                    }
    }

private:
    CU &PSB;
    DLU &L1;
    L2 &L2;
};

#endif // VU_HPP