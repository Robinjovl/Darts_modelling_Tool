#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <type_traits>
#include <algorithm>
#include <ostream>
#include <array>
#include <valarray>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <limits>
#include <iomanip>
#include <Eigen/Dense>
#define EQUALITY_TOLERANCE 1.E-10 //TODO: move to global.h

namespace linalg
{
	using std::abs;
	typedef int index_t;
	typedef double value_t; //TODO: move to global.h

	template<typename T>
	class Matrix
	{
	public:
		typedef T Type;
		int M, N;
	public:
		std::valarray<T> values;
		std::gslice g;

		Matrix() {};
		Matrix(const index_t _M, const index_t _N) : M(_M), N(_N), values(M * N)
		{
			std::fill_n(&values[0], M * N, 0.0);
		};
		Matrix(const Matrix<T>& m)
		{
			(*this) = m;
		};
		Matrix(const std::valarray<T>& v, const index_t _M, const index_t _N) : M(_M), N(_N), values(v) {};
		Matrix<T>& operator=(const Matrix<T>& m)
		{
			M = m.M;
			N = m.N;
			values = m.values;
			return *this;
		};
		Matrix<T>& operator+=(const Matrix<T>& m)
		{
			values += m.values;
			return *this;
		}
		Matrix<T>& operator-=(const Matrix<T>& m)
		{
			values -= m.values;
			return *this;
		}
		inline index_t getIndex(index_t i, index_t j) const
		{
			return i * N + j;
		};
		T operator()(const index_t i, const index_t j) const
		{
			return this->values[getIndex(i, j)];
		};
		T& operator()(const index_t i, const index_t j)
		{
			return this->values[getIndex(i, j)];
		};
		std::valarray<T> operator()(const std::size_t start, std::valarray<std::size_t> sizes, std::valarray<std::size_t> strides) const
		{
			const auto s = std::gslice(start, sizes, strides);
			return values[s];
		};
		std::gslice_array<T> operator()(const std::size_t start, std::valarray<std::size_t> sizes, std::valarray<std::size_t> strides)
		{
			g = std::gslice(start, sizes, strides);
			//std::cout << *this << std::endl;
			return values[g];
		};
		Matrix<T> transpose() const
		{
			Matrix<T> result (N, M);
			for (index_t i = 0; i < M; i++)
				for (index_t j = 0; j < N; j++)
					result(j, i) = (*this)(i, j);
			return result;
		};
		void transposeInplace()
		{
			(*this) = transpose();
		}
		void write_in_file(const std::string a) const
		{
			std::ofstream out(a.c_str(), std::ofstream::out);
			out << *this;
			out.close();
		}
		void set_diagonal(const std::valarray<T>& v)
		{
			for (index_t i = 0; i < M; i++)
				values[i * N + i] = v[i];
		}
		bool inv();
		bool svd(Matrix<T>& vc, std::valarray<T>& w);
	};
	template<typename T>
	Matrix<T> operator-(const Matrix<T>& m)
	{
		Matrix<T> result (-m.values, m.M, m.N);
		return result;
	}
	template<typename T>
	Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2)
	{
		assert(m1.values.size() == m2.values.size());
		Matrix<T> result (m1.values + m2.values, m1.M, m1.N);
		return result;
	}
	template<typename T>
	Matrix<T> operator-(const Matrix<T>& m1, const Matrix<T>& m2)
	{
		assert(m1.values.size() == m2.values.size());
		Matrix<T> result(m1.values - m2.values, m1.M, m1.N);
		return result;
	}
	template<typename T>
	Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2)
	{
		assert(m1.N == m2.M);
		Matrix<T> result (m1.M, m2.N);
		for (index_t i = 0; i < m1.M; i++)
			for (index_t j = 0; j < m2.N; j++)
			{
				for (index_t n = 0; n < m1.N; n++)
				{
					result(i, j) += m1(i, n) * m2(n, j);
				}
			}
		return result;
	}
	template<typename T>
	Matrix<T> operator*(const Matrix<T>& m, const T x)
	{
		Matrix<T> result (x * m.values, m.M, m.N);
		return result;
	}
	template<typename T>
	Matrix<T> operator*(const T x, const Matrix<T>& m)
	{
		return m * x;
	}
	template<typename T>
	Matrix<T> operator/(const Matrix<T>& m, const T x)
	{
		return m * (1 / x);
	}
	template<typename T,
		typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
		bool operator==(const Matrix<T>& m1, const Matrix<T>& m2)
	{
		//return (m1.values == m2.values).min();
		return std::abs(m1.values - m2.values).max() < EQUALITY_TOLERANCE;
	}
	template<typename T>
	bool operator!=(const Matrix<T>& m1, const Matrix<T>& m2)
	{
		return !(m1 == m2);
	}
	template<typename T,
		typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
		inline std::ostream& operator<<(std::ostream& os, const Matrix<T>& m)
	{
		for (int i = 0; i < m.M; i++)
		{
			for (int j = 0; j < m.N; j++)
			{
				os << std::setprecision(std::numeric_limits<double>::max_digits10) << m(i, j) << "\t";
			}
			os << std::endl;
		}
		return os;
	}

	template<typename Block>
	Matrix<typename Block::Type> make_block_diagonal(const Block& block, const size_t rank)
	{
		const uint8_t MB = block.M;
		const uint8_t NB = block.N;
		Matrix<typename Block::Type> result (rank * MB, rank * NB);
		for (uint8_t i = 0; i < rank; i++)
		{
			result(i * (NB + MB * rank * NB),
				{ MB, NB }, { rank * NB, 1 }) = block.values;
		}
		return result;
	};
	template<typename T>
	Matrix<T> outer_product(const Matrix<T>& m1, const Matrix<T>& m2)
	{
		Matrix<T> result (m1.M, m2.N);
		for (index_t i = 0; i < m1.M; i++)
			for (index_t j = 0; j < m2.N; j++)
			{
				result(i, j) = m1(i, 0) * m2(0, j);
			}
		return result;
	}

	template <typename T>
	inline double epsilon(const T& v)
	{
		T ep = std::numeric_limits<T>::epsilon() * 1e4;
		return v > 1.0 ? v * ep : ep;
	}
	template <class T>
	inline T sign(T a, T b)
	{
		return (b >= T(0) ? abs(a) : -abs(a));
	}
	template <typename T>
	bool Matrix<T>::inv()
	{
		assert(this->M == this->N);

		T* ptr = &this->values[0];
		T max_column, tmp;
		index_t i, j, k, i_max, i_flat, j_flat;

		std::valarray<index_t> row_index(this->M);
		std::iota(std::begin(row_index), std::end(row_index), 0);

		for (i = 0; i < this->M; i++)
		{
			// looking for a maximum in each column
			i_flat = i * this->N;
			max_column = abs(ptr[i_flat + i]);
			i_max = i;
			for (j = i + 1; j < this->M; j++)
			{
				tmp = abs(ptr[j * this->N + i]);
				if (tmp > max_column)
				{
					i_max = j;
					max_column = tmp;
				}
			}

			//if (max_column < epsilon(a))
			//{
			//	return false;
			//}

			// putting maximum to diagonal
			if (i_max != i)
			{
				std::swap(row_index[i_max], row_index[i]);
				j_flat = i_max * this->N;
				for (j = 0; j < this->N; j++)
				{
					std::swap(ptr[i_flat + j], ptr[j_flat + j]);
				}
			}

			// divide by maximum value
			tmp = T(1) / ptr[i_flat + i];
			ptr[i_flat + i] = T(1);
			for (j = 0; j < this->N; j++)
			{
				ptr[i_flat + j] *= tmp;
			}

			// elimination
			for (j = 0; j < this->M; j++)
			{
				if (j != i)
				{
					j_flat = j * this->N;
					tmp = ptr[j_flat + i];
					ptr[j_flat + i] = T(0);
					for (k = 0; k < this->N; k++)
					{
						ptr[j_flat + k] -= tmp * ptr[i_flat + k];
					}
				}
			}
		}
		// swapping columns back
		for (j = 0; j < this->N; j++)
		{
			if (j != row_index[j])
			{
				for (k = j + 1; j != row_index[k]; k++) {}

				for (i = 0; i < this->M; i++)
					std::swap(ptr[i * this->N + j], ptr[i * this->N + k]);
				std::swap(row_index[j], row_index[k]);
			}
		}
		return true;
	}
	template <typename T>
	bool Matrix<T>::svd(Matrix<T>& vc, std::valarray<T>& w)
	{
		// create matrix for Eigen with M rows and N cols and pointer to values (without a copy) 
		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > eigen_matrix(&this->values[0], this->M, this->N);

		// perform SVD
		Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV>
			svd(eigen_matrix);

		const Eigen::MatrixXd U = svd.matrixU();
		const Eigen::MatrixXd V = svd.matrixV();
		const Eigen::VectorXd S = svd.singularValues();

		// prepare output arrays
		size_t m = M;
		size_t n = N;
		if (vc.M != n || vc.N != n)
			vc = Matrix<T>(n, n);
		if (w.size() != n)
			w.resize(n);

		// copy from Eigen SVD to output arrays
		std::fill_n(&vc.values[0], vc.values.size(), 0.0);
		std::fill_n(&w[0], w.size(), 0.0);
		std::fill_n(&this->values[0], this->values.size(), 0.0);
		int n_max = this->N > this->M ? this->N : this->M;
		for (int i = 0; i < n_max; i++) {
			if (i < S.rows())
				w[i] = S(i);
			for (int j = 0; j < n_max; j++) {
				if (i < vc.M && j < vc.N)
					vc(i, j) = i < V.rows() && j < V.cols() ? V(i, j) : 0.;
				if (i < this->M && j < this->N)
					this->operator()(i, j) = i < U.rows() && j < U.cols() ? U(i, j) : 0.;
			}
		}
		return true;
	}
}

#endif /* MATRIX_H_ */
