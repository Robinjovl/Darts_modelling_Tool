#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <type_traits>
#include <algorithm>
#include <ostream>
#include <array>
#include <numeric>
#include <valarray>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <iomanip>
#include <Eigen/Dense>
#define EQUALITY_TOLERANCE 1.E-10

namespace linalg
{
  using std::abs;
  typedef int index_t;
  typedef double value_t; //TODO: move to global.h

  //!  Dense matrix class
  /*!
	This class contains slicing and other matrix operators and used in discretizer
  */
  template<typename T>
  class Matrix
  {
  public:
	typedef T Type; // datatype for the matrix values array
	int M; // number of columns
	int N; // number of rows
  public:
	std::valarray<T> values;
	std::gslice g;

	Matrix() {};
	Matrix(const index_t _M, const index_t _N) : M(_M), N(_N), values(M* N)
	{
	  if (M * N > 0) std::fill_n(&values[0], M * N, 0.0);
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
	//! extracts a sub-matrix from a given matrix
	/*!
	  \param start 1D index of starting position of etracting slice
	  \param sizes dimensions of sub-matrix
	  \param strides should be 1 if extract a row, and {M,1} if extract a column
	  \return sub-matrix
	*/
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
	  Matrix<T> result(N, M);
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
	inline bool is_nan() const
	{
	  for (const auto& val : values)
		if (val != val)
		  return true;
	  return false;
	};
	bool inv();
	bool svd(Matrix<T>& vc, std::valarray<T>& w);
  protected:
  };
  template<typename T>
  Matrix<T> operator-(const Matrix<T>& m)
  {
	Matrix<T> result(-m.values, m.M, m.N);
	return result;
  }
  template<typename T>
  Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2)
  {
	assert(m1.values.size() == m2.values.size());
	Matrix<T> result(m1.values + m2.values, m1.M, m1.N);
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
	Matrix<T> result(m1.M, m2.N);
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
	Matrix<T> result(x * m.values, m.M, m.N);
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
	const size_t MB = block.M;
	const size_t NB = block.N;
	Matrix<typename Block::Type> result(rank * MB, rank * NB);
	for (size_t i = 0; i < rank; i++)
	{
	  result(i * (NB + MB * rank * NB),
		{ MB, NB }, { rank * NB, 1 }) = block.values;
	}
	return result;
  };
  template<typename T>
  Matrix<T> outer_product(const Matrix<T>& m1, const Matrix<T>& m2)
  {
	Matrix<T> result(m1.M, m2.N);
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
  /*template <typename T>
  bool Matrix<T>::svd(Matrix<T>& vc, std::valarray<T>& w)
  {
	  Eigen::MatrixXd eigen_matrix = Eigen::MatrixXd(this->M, this->N);

	  // copy matrix to Eigen class
	  for (int i = 0; i < this->M; i++){
		  for (int j = 0; j < this->N; j++){
			  eigen_matrix(i, j) = this->operator()(i, j);
		  }
	  }

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
	  for (int i = 0; i < n_max; i++){
		  if (i < S.rows())
			  w[i] = S(i);
		  for (int j = 0; j < n_max; j++){
			  if (i < vc.M && j < vc.N)
				  vc(i, j) = i < V.rows() && j < V.cols() ? V(i, j) : 0.;
			  if (i < this->M && j < this->N)
				  this->operator()(i, j) = i < U.rows() && j < U.cols() ? U(i, j) : 0.;
		  }
	  }
	  return true;
  }*/

  template <typename T>
  bool Matrix<T>::svd(Matrix<T>& vc, std::valarray<T>& w)
  {
	//printf("OLD SVD!\n");
	size_t flag, i, its, j, jj, k, l, nm;
	T c, f, h, s, x, y, z, tmp;
	size_t m = this->M;
	size_t n = this->N;

	//if (vc.M != n || vc.N != n)
	//	vc = Matrix<T>(n, n);
	//if (w.size() != n)
	//	w.resize(n);

	T* a = &values[0];
	T* v = &vc.values[0];
	std::valarray<T> rv1(n);

	T g(0), scale(0), anorm(0);
	for (i = 0; i < n; i++)
	{
	  l = i + 1;
	  rv1[i] = scale * g;
	  g = s = scale = T(0);
	  if (i < m)
	  {
		for (k = i; k < m; k++)
		  scale += abs(a[k * n + i]);

		if (scale > epsilon(scale))
		{
		  for (k = i; k < m; k++)
		  {
			tmp = a[k * n + i] /= scale;
			s += tmp * tmp;
		  }
		  f = a[i * n + i];
		  g = -sign(sqrt(s), f);
		  h = f * g - s;
		  a[i * n + i] = f - g;

		  for (j = l; j < n; j++)
		  {
			for (s = T(0), k = i; k < m; k++)
			  s += a[k * n + i] * a[k * n + j];
			f = s / h;
			for (k = i; k < m; k++)
			  a[k * n + j] += f * a[k * n + i];
		  }
		  for (k = i; k < m; k++)
			a[k * n + i] *= scale;
		}
	  }
	  w[i] = scale * g;
	  g = s = scale = T(0);
	  if (i < m && i != n - 1)
	  {
		for (k = l; k < n; k++)
		  scale += abs(a[i * n + k]);

		if (scale > epsilon(scale))
		{
		  for (k = l; k < n; k++)
		  {
			tmp = a[i * n + k] /= scale;
			s += tmp * tmp;
		  }
		  f = a[i * n + l];
		  g = -sign(sqrt(s), f);
		  h = f * g - s;
		  a[i * n + l] = f - g;

		  for (k = l; k < n; k++)
			rv1[k] = a[i * n + k] / h;

		  for (j = l; j < m; j++)
		  {
			for (s = T(0), k = l; k < n; k++)
			  s += a[j * n + k] * a[i * n + k];
			for (k = l; k < n; k++)
			  a[j * n + k] += s * rv1[k];
		  }
		  for (k = l; k < n; k++)
			a[i * n + k] *= scale;
		}
	  }
	  anorm = std::max(anorm, abs(w[i]) + abs(rv1[i]));
	}

	for (i = n - 1; ; i--)
	{
	  if (i < n - 1)
	  {
		if (abs(g) > epsilon(g))
		{
		  for (j = l; j < n; j++)
			v[j * n + i] = (a[i * n + j] / a[i * n + l]) / g;
		  for (j = l; j < n; j++)
		  {
			for (s = T(0), k = l; k < n; k++)
			  s += a[i * n + k] * v[k * n + j];

			for (k = l; k < n; k++)
			  v[k * n + j] += s * v[k * n + i];
		  }
		}
		for (j = l; j < n; j++)
		  v[i * n + j] = v[j * n + i] = T(0);
	  }
	  v[i * n + i] = T(1);
	  g = rv1[i];
	  l = i;
	  if (i == 0)
		break;
	}

	for (i = std::min(m, n) - 1; ; i--)
	{
	  l = i + 1;
	  g = w[i];
	  for (j = l; j < n; j++)
		a[i * n + j] = T(0);
	  if (abs(g) > epsilon(g))
	  {
		g = T(1) / g;
		for (j = l; j < n; j++)
		{
		  for (s = T(0), k = l; k < m; k++)
			s += a[k * n + i] * a[k * n + j];

		  f = (s / a[i * n + i]) * g;

		  for (k = i; k < m; k++)
			a[k * n + j] += f * a[k * n + i];
		}
		for (j = i; j < m; j++)
		  a[j * n + i] *= g;

	  }
	  else
	  {
		for (j = i; j < m; j++)
		  a[j * n + i] = T(0);
	  }
	  ++a[i * n + i];
	  if (i == 0)
		break;
	}

	for (k = n - 1; ; k--)
	{
	  for (its = 1; its <= 30; its++)
	  {
		flag = 1;
		for (l = k; ; l--)
		{
		  nm = l - 1;
		  if (abs(rv1[l]) < epsilon(rv1[l]))
		  {
			flag = 0;
			break;
		  }
		  if (abs(w[nm]) < epsilon(w[nm]))
			break;
		  if (l == 0)
			break;
		}
		if (flag)
		{
		  c = T(0);
		  s = T(1);
		  for (i = l; i <= k; i++)
		  {
			f = s * rv1[i];
			rv1[i] = c * rv1[i];
			if (abs(f) < epsilon(f))
			  break;
			g = w[i];
			h = hypot(f, g);
			w[i] = h;
			h = T(1) / h;
			c = g * h;
			s = -f * h;
			for (j = 0; j < m; j++)
			{
			  y = a[j * n + nm];
			  z = a[j * n + i];
			  a[j * n + nm] = y * c + z * s;
			  a[j * n + i] = z * c - y * s;
			}
		  }
		}
		z = w[k];
		if (l == k)
		{
		  if (z < T(0))
		  {
			w[k] = -z;
			for (j = 0; j < n; j++)
			  v[j * n + k] = -v[j * n + k];
		  }
		  break;
		}

		if (its == 30)
		  return false;

		x = w[l];
		nm = k - 1;
		y = w[nm];
		g = rv1[nm];
		h = rv1[k];
		f = ((y - z) * (y + z) + (g - h) * (g + h)) / (T(2) * h * y);
		g = hypot(f, T(1));
		f = ((x - z) * (x + z) + h * ((y / (f + sign(g, f))) - h)) / x;
		c = s = T(1);
		for (j = l; j <= nm; j++)
		{
		  i = j + 1;
		  g = rv1[i];
		  y = w[i];
		  h = s * g;
		  g = c * g;
		  z = hypot(f, h);
		  rv1[j] = z;
		  c = f / z;
		  s = h / z;
		  f = x * c + g * s;
		  g = g * c - x * s;
		  h = y * s;
		  y *= c;
		  for (jj = 0; jj < n; jj++)
		  {
			x = v[jj * n + j];
			z = v[jj * n + i];
			v[jj * n + j] = x * c + z * s;
			v[jj * n + i] = z * c - x * s;
		  }
		  z = hypot(f, h);
		  w[j] = z;
		  if (abs(z) > epsilon(z))
		  {
			z = 1.0 / z;
			c = f * z;
			s = h * z;
		  }
		  f = c * g + s * y;
		  x = c * y - s * g;
		  for (jj = 0; jj < m; jj++)
		  {
			y = a[jj * n + j];
			z = a[jj * n + i];
			a[jj * n + j] = y * c + z * s;
			a[jj * n + i] = z * c - y * s;
		  }
		}
		rv1[l] = 0.0;
		rv1[k] = f;
		w[k] = x;
	  }
	  if (k == 0)
		break;
	}
	return true;
  }
}

#endif /* MATRIX_H */
