#pragma once

#include "linalg/matrix.h"
#include <vector>
#include <valarray>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>    // std::stable_sort
#include <numeric>      // std::iota
#include <cctype>
#include <locale>
#include <iterator>
#include "mesh/mesh.h"

namespace utils
{
	using linalg::index_t;
	using linalg::value_t;

	using mesh::ND;

	static inline std::pair<double, double> interpolateCoordinates(std::vector<double>& edge, double z_edge) {
		double x_coord = edge[0];
		double y_coord = edge[1];

		// Computes the interpolated (x,y) coordinates
		if (edge[2] != edge[5]){
			x_coord = (z_edge - edge[2]) / (edge[5] - edge[2]) * (edge[3] - edge[0]) + edge[0];
			y_coord = (z_edge - edge[2]) / (edge[5] - edge[2]) * (edge[4] - edge[1]) + edge[1];
		}
		std::pair<double, double> coords{ x_coord, y_coord };
		return coords;
	}

	static inline size_t from3Dto1DIndex(size_t ix, size_t iy, size_t iz, size_t nx, size_t ny, size_t nz) {
		// get the index of an element in a 3d matrix in a corresponding 1d array 
		return (iz + iy * nz + ix * (ny * nz));
	}

	// https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring
	// trim from start (in place)
	static inline void ltrim(std::string &s) {
		s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
			return !std::isspace(ch);
		}));
	}
	// trim from end (in place)
	static inline void rtrim(std::string &s) {
		s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
			return !std::isspace(ch);
		}).base(), s.end());
	}
	// trim from both ends (in place)
	static inline void trim(std::string &s) {
		ltrim(s);
		rtrim(s);
	}
	// trim from start (copying)
	static inline std::string ltrim_copy(std::string s) {
		ltrim(s);
		return s;
	}
	// trim from end (copying)
	static inline std::string rtrim_copy(std::string s) {
		rtrim(s);
		return s;
	}
	// https://stackoverflow.blog/2019/10/11/c-creator-bjarne-stroustrup-answers-our-top-five-c-questions/
	// split
	template<typename Delim>
	std::string get_word(std::istream& ss, Delim d)
	{
		std::string word;
		for (char ch; ss.get(ch); )    // skip delimiters
			if (!d(ch)) {
				word.push_back(ch);
				break;
			}
		for (char ch; ss.get(ch); )    // collect word
			if (!d(ch))
				word.push_back(ch);
			else
				break;
		return word;
	}
	std::vector<std::string> inline split(const std::string& s, const std::string& delim)
	{
		std::stringstream ss(s);
		auto del = [&](char ch) { for (auto x : delim) if (x == ch) return true; return false; };

		std::vector<std::string> words;
		for (std::string w; (w = get_word(ss, del)) != ""; ) words.push_back(w);
		return words;
	}
	index_t inline count_lines(std::string filename)
	{
		std::ifstream infile(filename);
		index_t lines_count = static_cast<index_t>(std::count(std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>(), '\n'));
		infile.close();
		return lines_count;
	}

	template <typename T>
	void inline parse_value(T &value, const std::string &word, bool &break_flag)
	{
		value = 0;
		try { 
			value = (T) std::stod(word);
			//if (T == double)
			//	buf = std::stod(word);
			//else if (T == float)
			//	buf = std::stof(word);
			//else if (T == int)
			//	buf = std::stoi(word);
			//else if (T == long)
			//	buf = std::stol(word);
			//else
			//	throw std::runtime_error("unknown type in parsing " + word);
		}
		catch (const std::invalid_argument& ia) { 
			std::cerr << "Invalid argument: " << ia.what() << " in word: " << word << '\n';
			break_flag = true;
		}
	}

	// reads the array defined by 'keyword' from 'filename' into 'vec'
	// num_values - limit the number of values will be read (used for SPECGRID keyword to avoid strtod failure for char parameter)
	template <typename T>
	void load_single_keyword(std::vector<T> &res, const std::string filename, const std::string keyword, const int num_values=-1)
	{
		index_t lines_num = count_lines(filename);
		std::ifstream infile(filename);
		std::vector<value_t> b;
		value_t buf;
		bool read_data_mode = false;
		bool break_flag = false;
		res.reserve(6 * lines_num);
		b.reserve(6);

		std::string line, first_word;
		while (std::getline(infile, line))
		{
			first_word = "";
			trim(line);

			if (!read_data_mode) // search keyword
			{
				const auto& words = split(line, " ");
				if (words.size()) first_word = words[0];

				if (first_word == keyword)
				{
					read_data_mode = 1;
					printf("Reading %s from %s\n", keyword.c_str(), filename.c_str());
					continue;
				}

				// skip INCLUDE // if (line == "INCLUDE")
			}

			if (!read_data_mode || line.size() == 0 || line[0] == '#' || (line[0] == '-' && line[1] == '-'))
				continue;

			//remove inline comments, for line like this:  5.6 6.7 --comment
			size_t idx = line.find_first_of("-");
			if (idx != std::string::npos && idx + 1 < line.size() && line.at(idx + 1) == '-')
				line = line.substr(0, idx);

			auto s1 = split(line, " \t"); // space and tabs can be delimiters
			for (const auto& word : s1)
			{
				// break when slash found
				if (word == "/") break;

				if (word.find('*') != std::string::npos) {
					auto s2 = split(word, "*");
					for (int m = 0; m < std::stoi(s2[0]); m++)
					{
						parse_value(buf, s2[1], break_flag);
						if (break_flag) {
							return;
						}
						b.push_back(buf);
						if (b.size() == num_values)
							break;
					}
				}
				else {
					parse_value(buf, word, break_flag);
					if (break_flag) {
						return;
					}
					b.push_back(buf);
					if (b.size() == num_values)
						break;
				}
			}

			res.insert(res.end(), b.begin(), b.end());
			b.clear();

			// break when slash found
			if (line.find('/') != -1) break;
		}

		infile.close();
		printf("Reading %s from %s finished. %zu values has been read.\n", keyword.c_str(), filename.c_str(), res.size());
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
			[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

		return idx;
	}

	template <typename T>
	inline std::valarray<T> get_valarray_from_array(const std::array<T, ND> a)
		{
			return std::valarray<value_t>(a.data(), a.size());
		}

}// namespace utils
