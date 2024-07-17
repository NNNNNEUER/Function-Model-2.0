#ifndef RDTU_HPP
#define RDTU_HPP

#include "DLU.hpp"

class RDTU
{
public:
    RDTU(DLU &DLU, size_t RMB_data_space, size_t MMB_data_space)
        : L1_cache(DLU), m_RMB(RMB_data_space), m_MMB(MMB_data_space){};

    // load into RMB as a (Co0,Kh*Kw) matrix,
    // there is essentially no data layout transformation here
    void loadWeight(size_t Weight_L1_addr,
                    size_t Co0, size_t Kh, size_t Kw, size_t Ci0)
    {
#define OFFSETRDTU(oi, ki, kj) (oi * Kh * Kw + ki * Kw + kj)
        for (size_t oi = 0; oi < Co0; ++oi)
            for (size_t ki = 0; ki < Kh; ++ki)
                for (size_t kj = 0; kj < Kw; ++kj)
                    for (size_t c = 0; c < Ci0; ++c)
                        m_RMB[OFFSETRDTU(oi, ki, kj) * Ci0 + c] =
                            L1_cache.getWeightCacheLine(Weight_L1_addr,
                                                        oi, ki, kj,
                                                        Kh, Kw,
                                                        Ci0)[c];
    }

    // load into MMB as a (Co0) vector
    void loadBias(size_t Bias_L1_addr, size_t Co0)
    {
        for (size_t oi = 0; oi < Co0; ++oi)
            m_MMB[oi] = L1_cache.getBiasVector(Bias_L1_addr, Co0)[oi];
    }

    // Get the SubMatrix(N0*K0) in row Ni and column Kj of
    // the matrix (Co0, Kh*Kw*Ci0) = (N1*N0,K1*K0) in RMB
    std::vector<int8_t> getWeightSubMatrix(size_t i, size_t j,
                                           size_t N, size_t K,
                                           size_t N0, size_t K0)
    {
        std::vector<int8_t> ret(N0 * K0);
        for (size_t n = 0; n < N0; n++)
        {
            auto N_i = i * N0 + n;
            if (N_i < N)
                for (size_t k = 0; k < K0; k++)
                {
                    auto K_j = j * K0 + k;
                    if (K_j < K)
                        ret[n * K0 + k] = m_RMB[N_i * K + K_j];
                }
        }

        return ret;
    }

    // Get the ith subvector (N0) of the vector (N = Co0) in MMB
    std::vector<int8_t> getBiasSubVector(size_t i, size_t N0, size_t N)
    {
        std::vector<int8_t> ret(N0);
        for (size_t n = 0; n < N0; n++)
        {
            auto N_i = i * N0 + n;
            if (N_i < N)
                ret[n] = m_MMB[N_i];
        }

        return ret;
    }

private:
    DLU &L1_cache;
    std::vector<int8_t> m_RMB;
    std::vector<int8_t> m_MMB;
};

#endif // RDTU_HPP