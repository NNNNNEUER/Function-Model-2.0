#ifndef LDTU_HPP
#define LDTU_HPP

#include "CubeCache.hpp"
#include "DLU.hpp"

class LDTU
{
public:
    LDTU(DLU &DLU, size_t num_cacheline) : L1_cache(DLU), m_LMB(num_cacheline){};

    // Matrix im2col to (H0*W0,Kh*Kw) in LMB from (H0+ kh-1,W0+ kw-1) subPaddedFeature in L1
    void load_im2col(size_t Feature_L1_addr, size_t H0, size_t W0, size_t Kh, size_t Kw)
    {
#define OFFSETLDTU(i, j, ki, kj) ((i * W0 + j) * (Kh * Kw) + (ki * Kw + kj))
        // row
        for (size_t i = 0; i < H0; i++)
        {
            for (size_t j = 0; j < W0; j++)
            {
                // col
                for (size_t ki = 0; ki < Kh; ki++)
                {
                    for (size_t kj = 0; kj < Kw; kj++)
                    {
                        m_LMB[OFFSETLDTU(i, j, ki, kj)] =
                            L1_cache.getFeatureCacheLine(Feature_L1_addr, i + ki, j + kj, W0, Kw);
                    }
                }
            }
        }
    }

    // Get M0*C0 elements
    // the matrix (H0*W0,Kh*Kw) in LMB
    std::vector<CubeCacheLine> getCacheLine(size_t M1, size_t K1,
                                            size_t M0, size_t K0, size_t M, size_t K)
    {
        size_t num_K = K0 / 16;
        std::vector<CubeCacheLine> cacheLine(M0 * num_K);
        for (size_t i = 0; i < M0; i++)
        {
            auto M_i = M1 * M0 + i;
            if (M_i < M)
                for (size_t j = 0; j < num_K; j++)
                {
                    auto K_j = K1 * num_K + j;
                    cacheLine[i * num_K + j] = m_LMB[M_i * K / K0 + K_j];
                }
            // cacheLine[i] = m_LMB[M_i * K / K0 + K1];
        }
        return cacheLine;
    }

private:
    DLU &L1_cache;
    CubeCache m_LMB;
};

#endif // LDTU_HPP