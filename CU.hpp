#ifndef CU_HPP
#define CU_HPP

#include "LDTU.hpp"
#include "RDTU.hpp"
#include <cassert>
#include <iostream>

class CU
{
public:
    CU(LDTU &LDTU, RDTU &RDTU, size_t PSB_data_space)
        : LMB(LDTU), RMB(RDTU), m_PSB(PSB_data_space) {}

    // inquire PSB
    std::vector<int8_t> &getPSB() { return m_PSB; }

    // clear PSB
    void clearPSB() { m_PSB.clear(); }

    // matmul
    void matmulCtrl(bool Bias_en, bool Psum_en,
                    size_t mi, size_t nj, size_t kk,
                    size_t M, size_t N, size_t K,
                    size_t M0, size_t N0, size_t K0)
    {
        // M : H0 * W0
        // K : Kh * Kw
        // N : Co0

        // M1 is the index with [0, CEIL(H0 * W0, M0))
        // K1 is the index with [0, CEIL(Kh * Kw, K0))
        // N1 is the index with [0, CEIL(Co0, N0))

        // LMB : (H0 * W0) * (Kh * Kw) = M * K = (M1 * M0) * (K1 * K0)
        // RMB : (Co0) * (Kh * Kw) = N * K = (N1 * N0) * (K1 * K0)
        // PSB : (H0 * W0) * Co0  This is still HWC

        if (Bias_en && Psum_en)
        {
            std::cerr << "Error! Bias_en and Psun_en are both high!" << std::endl;
            assert(Bias_en && Psum_en);
        }
        else if (Bias_en && !Psum_en)
        {
            // TODO: bias
        }
        else if (!Bias_en && Psum_en)
        {
            // LDTU
            // to perform legit multiplication, size(LMB_line) = M0 * K0
            std::vector lhs = LMB.getFeatureSubMatrix(mi, kk, M, K, M0, K0);

            // RDTU
            // to perform legit multiplication, size(RMB_line) = N0 * K0
            std::vector rhs = RMB.getWeightSubMatrix(nj, kk, N, K, N0, K0);

            // Note that MatMul requires right to be transposed
            std::vector<int8_t> result(M0 * N0);
            for (size_t m = 0; m < M0; m++)
                for (size_t n = 0; n < N0; n++)
                    for (size_t k = 0; k < K0; k++)
                        result[m * N0 + n] += lhs[m * K0 + k] * rhs[n * K0 + k];

            // PSB: M * N1*N0
            // There are N1 cachelines in a row,
            // storing N0 channels (if N < N0 then there will be padding)
            for (size_t m = 0; m < M0; m++)
            {
                auto M_i = mi * M0 + m;
                if (M_i < M)
                    for (size_t n = 0; n < N0; n++)
                    {
                        auto N_j = nj * N0 + n;
                        if (N_j < N)
                            m_PSB[M_i * N + N_j] = result[m * N0 + n];
                    }
            }
        }
        else
        {
            std::cerr << "Do nothing! Bias_en and Psun_en are both low!" << std::endl;
            return;
        }
    }

private:
    LDTU &LMB;
    RDTU &RMB;
    std::vector<int8_t> m_PSB;
};

#endif // CU_HPP