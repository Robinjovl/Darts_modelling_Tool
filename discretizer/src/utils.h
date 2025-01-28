#pragma once

#include "elem.h"
#include "linalg/matrix.h"
#include "logger.h"
#include <cctype>
#include <fstream>
#include <numeric> // std::iota
#include <string>
#include <valarray>
#include <vector>

namespace utils {
using linalg::index_t;
using linalg::value_t;

using mesh::ND;

void trim(std::string &s);

// https://stackoverflow.blog/2019/10/11/c-creator-bjarne-stroustrup-answers-our-top-five-c-questions/
// split
template <typename Delim> std::string get_word(std::istream &ss, Delim d) {
  std::string word;
  for (char ch; ss.get(ch);) // skip delimiters
    if (!d(ch)) {
      word.push_back(ch);
      break;
    }
  for (char ch; ss.get(ch);) // collect word
    if (!d(ch))
      word.push_back(ch);
    else
      break;
  return word;
}

std::vector<std::string> split(const std::string &s, const std::string &delim);

index_t count_lines(std::string filename);

template <typename T>
void inline parse_value(T &value, const std::string &word, bool &break_flag) {
  value = 0;
  try {
    value = (T)std::stod(word);
    // if (T == double)
    //	buf = std::stod(word);
    // else if (T == float)
    //	buf = std::stof(word);
    // else if (T == int)
    //	buf = std::stoi(word);
    // else if (T == long)
    //	buf = std::stol(word);
    // else
    //	throw std::runtime_error("unknown type in parsing " + word);
  } catch (const std::invalid_argument &ia) {
    dis::logger.error("Invalid argument: {} in word: {}", ia.what(), word);
    break_flag = true;
  }
}

// reads the array defined by 'keyword' from 'filename' into 'vec'
// num_values - limit the number of values will be read (used for SPECGRID
// keyword to avoid strtod failure for char parameter)
template <typename T>
void load_single_keyword(std::vector<T> &res, const std::string filename,
                         const std::string keyword, const int num_values = -1) {
  index_t lines_num = count_lines(filename);
  std::ifstream infile(filename);
  std::vector<value_t> b;
  value_t buf;
  bool read_data_mode = false;
  bool break_flag = false;
  res.reserve(6 * lines_num);
  b.reserve(6);

  std::string line, first_word;
  while (std::getline(infile, line)) {
    first_word = "";
    trim(line);

    if (!read_data_mode) // search keyword
    {
      const auto &words = split(line, " ");
      if (words.size())
        first_word = words[0];

      if (first_word == keyword) {
        read_data_mode = 1;
        dis::logger.info("Reading {} from {}", keyword, filename);
        continue;
      }

      // skip INCLUDE // if (line == "INCLUDE")
    }

    if (!read_data_mode || line.size() == 0 || line[0] == '#' ||
        (line[0] == '-' && line[1] == '-'))
      continue;

    // remove inline comments, for line like this:  5.6 6.7 --comment
    size_t idx = line.find_first_of("-");
    if (idx != std::string::npos && idx + 1 < line.size() &&
        line.at(idx + 1) == '-')
      line = line.substr(0, idx);

    auto s1 = split(line, " \t"); // space and tabs can be delimiters
    for (const auto &word : s1) {
      // break when slash found
      if (word == "/")
        break;

      if (word.find('*') != std::string::npos) {
        auto s2 = split(word, "*");
        for (int m = 0; m < std::stoi(s2[0]); m++) {
          parse_value(buf, s2[1], break_flag);
          if (break_flag) {
            return;
          }
          b.push_back(buf);
          if (num_values >= 0 && b.size() == static_cast<size_t>(num_values))
            break;
        }
      } else {
        parse_value(buf, word, break_flag);
        if (break_flag) {
          return;
        }
        b.push_back(buf);
        if (num_values >= 0 && b.size() == static_cast<size_t>(num_values))
          break;
      }
    }

    res.insert(res.end(), b.begin(), b.end());
    b.clear();

    // break when slash found
    if (line.find('/') != std::string::npos)
      break;
  }

  infile.close();
  dis::logger.info("Reading {} from {} finished. {} values has been read.",
                   keyword, filename, res.size());
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return idx;
}

template <typename T>
inline std::valarray<T> get_valarray_from_array(const std::array<T, ND> a) {
  return std::valarray<value_t>(a.data(), a.size());
}

} // namespace utils
