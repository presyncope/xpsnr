#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class yuv_reader
{
public:
  yuv_reader(const std::string &path, int width, int height, int bitdepth, int yuvformat = 420)
  {
    m_file.open(path, std::ios::binary);

    if (!m_file.is_open())
    {
      std::cerr << "yuv_reader: \"" << path << "\" open failed" << std::endl;
      return;
    }

    if (yuvformat != 420 && yuvformat != 422 && yuvformat != 444)
    {
      std::cerr << "yuv_reader: unsupported yuv format " << yuvformat << std::endl;
      return;
    }

    uint32_t left_shift = ((bitdepth + 7u) >> 3u) - 1u;
    uint32_t luma_frame_size = (width * height) << left_shift;
    uint32_t chroma_width = (yuvformat == 444) ? width : ((width + 1) >> 1);
    uint32_t chroma_height = (yuvformat == 420) ? ((height + 1) >> 1) : height;
    uint32_t chroma_frame_size = (chroma_width * chroma_height) << left_shift;

    m_frame_size = luma_frame_size + chroma_frame_size * 2u;

    m_buffer.resize(m_frame_size);
  }

public:
  const uint32_t get_frame_size() const { return m_frame_size; }

  const uint8_t *read_next()
  {
    if (!m_file.is_open())
    {
      return nullptr;
    }

    m_file.read((char *)m_buffer.data(), m_frame_size);

    if ((std::size_t)m_file.gcount() < m_frame_size)
    {
      m_file.close();
      return nullptr;
    }

    return m_buffer.data();
  }

private:
  std::ifstream m_file;
  uint32_t m_frame_size;
  std::vector<uint8_t> m_buffer;
};