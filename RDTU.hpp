#ifndef RDTU_HPP
#define RDTU_HPP

#include "DLU.hpp"

class RDTU
{
public:
    RDTU(DLU &DLU, size_t num_cacheline, size_t mmb_num_cacheline)
        : L1_cache(DLU), m_RMB(num_cacheline), m_MMB(mmb_num_cacheline) {}

    // load into RMB as a (Co0,Kh*Kw) matrix,
    // there is essentially no data layout transformation here
    void load(size_t Weight_L1_addr, size_t Co0, size_t Kh, size_t Kw)
    {
        // essentially copy starts at L1_addr from L1 [0,m_Co0*m_Kw*Kh)
        // no loop needed, but left for interface abstraction
        // Structure: matrix of Co0 * (Kw*Kh), one cacheline per element
#define OFFSETRDTU(oi, ki, kj) (oi * Kh * Kw + ki * Kw + kj)
        for (size_t oi = 0; oi < Co0; ++oi)
        {
            for (size_t ki = 0; ki < Kh; ++ki)
            {
                for (size_t kj = 0; kj < Kw; ++kj)
                {
                    m_RMB[OFFSETRDTU(oi, ki, kj)] =
                        L1_cache.getWeightCacheLine(Weight_L1_addr, oi, ki, kj, Kh, Kw);
                }
            }
        }
    }

    void loadMMB(size_t Bias_L1_addr, size_t Co1)
    {
        m_MMB[Co1] = L1_cache.getBiasCacheLine(Bias_L1_addr, Co1);
    }

    // Get the elements(CubeCacheLine) in row i and column j of
    // the matrix (Co0,Kh*Kw) in RMB
    std::vector<CubeCacheLine> getCacheLine(size_t N1, size_t K1,
                                            size_t N0, size_t K0, size_t N, size_t K)
    {
        size_t num_K = K0 / 16;
        std::vector<CubeCacheLine> RMB_line(N0 * num_K);
        for (size_t i = 0; i < N0; i++)
        {
            auto N_i = N1 * N0 + i;
            if (N_i < N)
                for (size_t j = 0; j < num_K; j++)
                {
                    auto K_j = K1 * num_K + j;
                    RMB_line[i * num_K + j] = m_RMB[N_i * K / K0 + K_j];
                }
            // RMB_line[i] = m_RMB[N_i * K / K0 + K1];
        }
        return RMB_line;
    }

    CubeCacheLine getBias(size_t Co1)
    {
        return m_MMB[Co1];
    }

private:
    DLU &L1_cache;
    CubeCache m_RMB;
    CubeCache m_MMB;
};

#endif // RDTU_HPP