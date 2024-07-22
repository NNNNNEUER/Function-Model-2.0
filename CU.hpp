#ifndef CU_HPP
#define CU_HPP

#include "CubeCache.hpp"
#include "LDTU.hpp"
#include "RDTU.hpp"

class CU
{
public:
    CU(LDTU &LDTU, RDTU &RDTU, size_t size_cacheline) : LMB(LDTU), RMB(RDTU), m_PSB(size_cacheline) {}

    // clear PSB
    void clearPSB()
    {
        for (size_t i = 0; i < m_PSB.size(); i++)
        {
            m_PSB[i].clear();
        }
    }

    // inquire PSB
    CubeCache &getPSB()
    {
        return m_PSB;
    }

    void matmul(size_t M1, size_t K1, size_t N1,
                size_t M0, size_t K0, size_t N0, size_t M, size_t K, size_t N)
    {
        // M : H0 * W0
        // K : Kh * Kw * C0
        // N : Co0

        // M1 is the index with [0, CEIL(H0 * W0, C0))
        // K1 is the index with [0, CEIL(Kh * Kw * C0, K0))
        // N1 is the index with [0, CEIL(Co0, C0))

        // LMB : (H0 * W0) * (Kh * Kw) = M * K = (M1 * M0) * (K1 * K0)
        // RMB : (Co0) * (Kh * Kw) = N * K = N * (K1 * K0)
        // PSB : (H0 * W0) * N This is still HWC,
        // the last dimension has 16 channels to fill up a cacheline,
        // if Co0 < C0 (N < N0) then there will be padding

        // LDTU
        // to perform legit multiplication, size(LMB_line) = m_M0
        std::vector<CubeCacheLine> LMB_line = LMB.getCacheLine(M1, K1, M0, K0, M, K);

        // RDTU holds the transpose of the right matrix
        std::vector<CubeCacheLine> RMB_line = RMB.getCacheLine(N1, K1, N0, K0, N, K); // m_N0

        // Note that MatMul requires right to be transposed
        std::vector<CubeCacheLine> result = MatMul(LMB_line, RMB_line, M0, N0, K0);

        // PSB: (H0 * W0) * N
        for (size_t i = 0; i < M0; i++)
        {
            auto M_i = M1 * M0 + i;
            // m_N0 = C0
            // There are CEIL(m_N/m_N0) = ((m_N + m_n0-1) /m_N0) cachelines in a row,
            // storing Co0 channels (if Co0 < C0 (N < N0) then there will be padding)
            m_PSB[M_i * ((N + N0 - 1) / N0) + N1].bitwiseAddCacheLine(result[i]);
        }
    }

    void presetBias(size_t Co1, size_t M, size_t N)
    {
        for (size_t n = 0; n < N; n++)
        {
            for (size_t m = 0; m < M; m++)
            {
                m_PSB[m][n] = RMB.getBias(Co1)[n];
            }
        }
    }

private:
    LDTU &LMB;
    RDTU &RMB;
    CubeCache m_PSB;
};

#endif // CU_HPP