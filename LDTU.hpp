#ifndef LDTU_HPP
#define LDTU_HPP

#include "DLU.hpp"
#include <iostream>

class LDTU
{
public:
    LDTU(DLU &DLU, size_t LMB_data_space)
        : L1_cache(DLU), m_LMB(LMB_data_space){};

    // Matrix img2col to (H0*W0,Kh*Kw) in LMB from (H0+kh-1,W0+kw-1) subPaddedFeature in L1
    void load_im2col(size_t feature_L1_addr,
                     size_t H0, size_t W0, size_t C0,
                     size_t Kh, size_t Kw)
    {
#define OFFSETLDTU(i, j, ki, kj) ((i * W0 + j) * (Kh * Kw) + (ki * Kw + kj))
        // row
        for (size_t i = 0; i < H0; i++)
            for (size_t j = 0; j < W0; j++)
                // col
                for (size_t ki = 0; ki < Kh; ki++)
                    for (size_t kj = 0; kj < Kw; kj++)
                        // channel
                        for (size_t c = 0; c < C0; c++)
                        {
                            m_LMB[OFFSETLDTU(i, j, ki, kj) * C0 + c] =
                                L1_cache.getFeatureCacheLine(feature_L1_addr, i + ki, j + kj, W0, Kw, C0)[c];
                        }
    }
    // Get the SubMatrix(M0*K0) in row Mi and column Kj of
    // the matrix (H0*W0,Kh*Kw*C0) = (M1*M0,K1*K0) in LMB
    std::vector<int8_t> getFeatureSubMatrix(size_t i, size_t j,
                                            size_t M, size_t K,
                                            size_t M0, size_t K0)
    {
        std::vector<int8_t> ret(M0 * K0);
        for (size_t m = 0; m < M0; m++)
        {
            auto M_i = i * M0 + m;
            if (M_i < M)
                for (size_t k = 0; k < K0; k++)
                {
                    auto K_j = j * K0 + k;
                    if (K_j < K)
                        ret[m * K0 + k] = m_LMB[M_i * K + K_j];
                }
        }
        return ret;
    }

private:
    DLU &L1_cache;
    std::vector<int8_t> m_LMB;
};

#endif // LDTU_HPP