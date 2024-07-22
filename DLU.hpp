#ifndef DLU_HPP
#define DLU_HPP

#include "L2.hpp"
#include "CubeCache.hpp"

class DLU
{
public:
    DLU(L2 &L2, size_t num_cacheline) : L2_cache(L2), m_L1_cache(num_cacheline) {}

    // Load subfeatures of size (H0+kh-1, W0+Kw-1) centered at subblocks (H1,W1,C1)
    // to addr (Kh/2,Kw/2 padding in all four directions, out of bounds is 0), stored in L1
    void loadSubPaddedFeature(size_t Feature_L1_addr, size_t H1, size_t W1, size_t C1,
                              size_t Feature_L2_addr, size_t H, size_t W, size_t C,
                              size_t H0, size_t W0, size_t Kh, size_t Kw, size_t C0)
    {
        // Reminder: A subFeature in L1 is a (H0+Kh-1,W0+Kw-1) matrix,
        // where each element is a cacheline with C0 channels.
        // Iterate over all elements to load i <- [0, H0+Kh-1), j <- [0, W0+Kw-1)
        for (size_t i = 0; i < H0 + Kh - 1; i++)
        {
            for (size_t j = 0; j < W0 + Kw - 1; j++)
            {
// Calculate the offset of the current element's addr relative to L1
// row major storage is used
#define OFFSETDLUFeature(i, j) (i * (W0 + Kw - 1) + j)
                // Read elements from L2, temporarily load them one by one
                for (size_t c = 0; c < C0; c++)
                {
                    // Calculate the address of the current element in L2
                    size_t L2_addr_row = H1 * H0 + (i - Kh / 2);
                    size_t L2_addr_col = W1 * W0 + (j - Kw / 2);
                    size_t L2_addr_cha = C1 * C0 + c;
                    // Read elements from L2
                    // judge whether (im_row, im_col, im_cha) is out of bound
                    if (L2_addr_row < H && L2_addr_col < W && L2_addr_cha < C)
                    {
                        size_t L2_addr_offset =
                            L2_addr_row * W * C +
                            L2_addr_col * C +
                            L2_addr_cha;
                        m_L1_cache[Feature_L1_addr + OFFSETDLUFeature(i, j)][c] =
                            L2_cache.read(Feature_L2_addr + L2_addr_offset);
                    }
                    else
                    {
                        m_L1_cache[Feature_L1_addr + OFFSETDLUFeature(i, j)][c] = 0;
                    }
                }
            }
        }
    }

    // Load subweight of size (Co0,Kh,Kw) centered at subblocks (Co1,Ci1) into addr and store in L1
    void loadSubWeight(size_t Weight_L1_addr, size_t Co1, size_t Ci1,
                       size_t Weight_L2_addr, size_t Co, size_t Ci,
                       size_t Co0, size_t Kh, size_t Kw, size_t Ci0)
    {
        // Reminder:subWeight is a (Co0,Kh,Kw) matrix in L1,
        // where each element is a cacheline, containing Ci0 channels.
        // Iterate over all the elements to load
        // Co0 Co channels: oi <- [0, Co0)
        for (size_t oi = 0; oi < Co0; oi++)
        {
            // Iterate over each element of weight ki <- [0, Kh), kj <- [0, Kw)
            for (size_t ki = 0; ki < Kh; ki++)
            {
                for (size_t kj = 0; kj < Kw; kj++)
                {
// Calculate the offset of the current element's addr relative to L1
// row major storage is used
#define OFFSETDLUWeight(oi, ki, kj) (oi * Kh * Kw + ki * Kw + kj)
                    // Load elements from L2, temporarily Ci0 elements one by one
                    for (size_t ci = 0; ci < Ci0; ci++)
                    {
                        // Calculate the address of the current element in L2
                        size_t L2_addr_cho = Co1 * Co0 + oi;
                        size_t L2_addr_row = ki;
                        size_t L2_addr_col = kj;
                        size_t L2_addr_chi = Ci1 * Ci0 + ci;
                        // Read elements from L2
                        // judge whether (im_cho, im_chi) is out of bound
                        if (L2_addr_cho < Co && L2_addr_chi < Ci)
                        {
                            size_t L2_addr_offset =
                                L2_addr_cho * Kh * Kw * Ci +
                                L2_addr_row * Kw * Ci +
                                L2_addr_col * Ci +
                                L2_addr_chi;
                            m_L1_cache[Weight_L1_addr + OFFSETDLUWeight(oi, ki, kj)][ci] =
                                L2_cache.read(Weight_L2_addr + L2_addr_offset);
                        }
                        else
                        {
                            m_L1_cache[Weight_L1_addr + OFFSETDLUWeight(oi, ki, kj)][ci] = 0;
                        }
                    }
                }
            }
        }
    }

    void loadSubBias(size_t Bias_L1_addr, size_t Co1,
                     size_t Bias_L2_addr, size_t Co, size_t Co0)
    {
        for (size_t oi = 0; oi < Co0; oi++)
        {
            size_t L2_addr_cho = Co1 * Co0 + oi;
            if (L2_addr_cho < Co)
            {
                m_L1_cache[Bias_L1_addr + Co1][oi] = L2_cache.read(Bias_L2_addr + L2_addr_cho);
            }
            else
            {
                m_L1_cache[Bias_L1_addr + Co1][oi] = 0;
            }
        }
    }

    // interpret the memory from addr as a (H0+Kh-1,W0+Kw-1) Feature matrix
    // with each entry being a cacheline(C0 elements)
    CubeCacheLine getFeatureCacheLine(size_t Feature_L1_addr, size_t i, size_t j,
                                      size_t W0, size_t Kw)
    {
        return m_L1_cache[Feature_L1_addr + i * (W0 + Kw - 1) + j];
    }

    // interpret the memory from addr as a (Co0,Kh,Kw) Weight matrix
    // with each entry a cacheline(Ci0 elements)
    CubeCacheLine getWeightCacheLine(size_t Weight_L1_addr, size_t oi, size_t ki, size_t kj,
                                     size_t Kh, size_t Kw)
    {
        return m_L1_cache[Weight_L1_addr + oi * Kh * Kw + ki * Kw + kj];
    }

    CubeCacheLine getBiasCacheLine(size_t Bias_L1_addr, size_t Co1)
    {
        return m_L1_cache[Bias_L1_addr + Co1];
    }

private:
    L2 &L2_cache;
    CubeCache m_L1_cache;
};

#endif // DLU_HPP