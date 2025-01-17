
#include "linalg/matrix.h"
#include "utils.h"

using namespace std;

namespace utils
{
	using linalg::index_t;
	using linalg::value_t;
	
vector<string> split(const string& s, const string& delim)
	{
		std::stringstream ss(s);
		auto del = [&](char ch) { for (auto x : delim) if (x == ch) return true; return false; };

		std::vector<std::string> words;
		for (std::string w; (w = get_word(ss, del)) != ""; ) words.push_back(w);
		return words;
	}

	
// I commented those because they are unused.
// Indeed, static functions are only available in the current file
// and it wasn't called in it.

/*static inline std::pair<double, double> interpolateCoordinates(std::vector<double>& edge, double z_edge) {*/
	/*	double x_coord = edge[0];*/
	/*	double y_coord = edge[1];*/
	/**/
	/*	// Computes the interpolated (x,y) coordinates*/
	/*	if (edge[2] != edge[5]){*/
	/*		x_coord = (z_edge - edge[2]) / (edge[5] - edge[2]) * (edge[3] - edge[0]) + edge[0];*/
	/*		y_coord = (z_edge - edge[2]) / (edge[5] - edge[2]) * (edge[4] - edge[1]) + edge[1];*/
	/*	}*/
	/*	std::pair<double, double> coords{ x_coord, y_coord };*/
	/*	return coords;*/
	/*}*/
	/*static inline size_t from3Dto1DIndex(size_t ix, size_t iy, size_t iz, size_t nx, size_t ny, size_t nz) {*/
	/*	// get the index of an element in a 3d matrix in a corresponding 1d array */
	/*	return (iz + iy * nz + ix * (ny * nz));*/
	/*}*/
	/**/
	/*// https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring*/
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
	void trim(std::string &s) {
		ltrim(s);
		rtrim(s);
	}
	/*// trim from start (copying)*/
	/*static  std::string ltrim_copy(std::string s) {*/
	/*	ltrim(s);*/
	/*	return s;*/
	/*}*/
	/*// trim from end (copying)*/
	/*static  std::string rtrim_copy(std::string s) {*/
	/*	rtrim(s);*/
	/*	return s;*/
	/*}*/

index_t count_lines(std::string filename) {
  std::ifstream infile(filename);
  index_t lines_count =
      static_cast<index_t>(std::count(std::istreambuf_iterator<char>(infile),
                                      std::istreambuf_iterator<char>(), '\n'));
  infile.close();
  return lines_count;
}



}// namespace utils
