#ifndef L2_HPP
#define L2_HPP

#include <cstdint>
#include <vector>

class L2
{
public:
    L2(size_t size) : m_data(size) {}
    uint8_t read(size_t addr) { return m_data.at(addr); }
    void write(size_t addr, uint8_t data) { m_data.at(addr) = data; }

private:
    std::vector<uint16_t> m_data;
};

#endif // L2_HPP