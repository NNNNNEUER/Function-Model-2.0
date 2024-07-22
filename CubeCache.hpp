#ifndef CUBECACHE_HPP
#define CUBECACHE_HPP

#include <cstdint>
#include <vector>

constexpr size_t C0 = 16;

// CubeCacheLine is the fundamental unit of L1, LMB, RMB, MMB and PSB
class CubeCacheLine
{
public:
    CubeCacheLine() : m_data(C0) {} // CubeCacheLine is a int_8 vector of size 16

    uint8_t operator[](size_t index) const { return m_data.at(index); }
    uint8_t &operator[](size_t index) { return m_data.at(index); }

    void clear() { std::fill(m_data.begin(), m_data.end(), 0); }
    void bitwiseAddCacheLine(const CubeCacheLine &other)
    {
        for (size_t i = 0; i < C0; i++)
        {
            m_data[i] += other[i];
        }
    }

private:
    std::vector<uint8_t> m_data;
};

// L1, LMB, RMB, MMB and PSB are all CubeCache
class CubeCache
{
public:
    CubeCache(size_t size) : m_size(size), m_data(size) {}
    size_t size() const { return m_size; }

    CubeCacheLine operator[](size_t index) const { return m_data.at(index); }
    CubeCacheLine &operator[](size_t index) { return m_data.at(index); }

private:
    size_t m_size;
    std::vector<CubeCacheLine> m_data;
};

// lhs <- LMB,是im2col后的feature,
// rhs <- RMB,是weight的转置
std::vector<CubeCacheLine> MatMul(std::vector<CubeCacheLine> &lhs, std::vector<CubeCacheLine> &rhs,
                                  size_t M0, size_t N0, size_t K0)
{
    size_t num_N = N0 / C0;
    size_t num_M = M0 / C0;
    size_t num_K = K0 / C0;
    std::vector<CubeCacheLine> result(M0 * num_N);

    for (size_t mi = 0; mi < num_M; mi++)
    {
        for (size_t nj = 0; nj < num_N; nj++)
        {
            for (size_t i = 0; i < C0; i++)
            {
                for (size_t j = 0; j < C0; j++)
                {
                    for (size_t k = 0; k < C0; k++)
                    {
                        result[(mi * num_N + nj) * C0 + i][j] += lhs[mi * C0 + i][k] * rhs[nj * C0 + j][k]; // 注意 right[**j**][**k**]
                    }
                }
            }
        }
    }
    return result;
}

#endif // CUBECACHE_HPP