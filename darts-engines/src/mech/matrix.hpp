#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <vector>
#include <type_traits>
#include <algorithm>
#include <ostream>
#include <array>
#include <valarray>
#include <assert.h>
#include "globals.h"
#include <iostream>
#include <limits>
#include <iomanip>
#define EQUALITY_TOLERANCE 1.E-10

namespace linalg
{
	using std::abs;
	
	template<typename T>
	class Matrix
	{
	public:
		typedef T Type;
		index_t M, N;
	public:
		std::valarray<T> values;
		std::gslice g;

		Matrix() {};
		Matrix(const index_t _M, const index_t _N) : M(_M), N(_N), values(M * N)
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
		inline bool is_nan() const 
		{
			for (const auto& val : values)
				if (val != val)
					return true;
			return false;
		};
		bool inv();
		bool lu(std::valarray<size_t>& ri, T* pDet);
		bool svd(Matrix<T>& vc, std::valarray<T>& w);
		T det() const;
		bool eigen(std::valarray<T>& rev, std::valarray<T>& iev) const;
	protected:
		void balanc(Matrix<T>& v, bool eivec);
		bool hqr2(std::valarray<T>& d, std::valarray<T>& e, Matrix<T>& v, bool eivec);
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
		const size_t MB = block.M;
		const size_t NB = block.N;
		Matrix<typename Block::Type> result (rank * MB, rank * NB);
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
		Matrix<T> result (m1.M, m2.N);
		for (index_t i = 0; i < m1.M; i++)
			for (index_t j = 0; j < m2.N; j++)
			{
				result(i, j) = m1(i, 0) * m2(0, j);
			}
		return result;
	}

	// taken from https://www.techsoftpl.com/matrix/doc/
	template <typename T>
	inline double epsilon(const T& v)
	{
		T ep = std::numeric_limits<T>::epsilon() * 1e4;
		return v > 1.0 ? v * ep : ep;
	}
	template <typename T>
	bool Matrix<T>::inv()
	{
		assert(M == N);
		size_t n = M;
		T *pv = &this->values[0];
		size_t i, j, k, ipos, kpos;

		std::valarray<size_t> ri(n);
		for (i = 0; i < n; i++)
			ri[i] = i;

		for (k = 0; k < n; k++)
		{
			double ta, tb;
			T a(0);

			kpos = k * n;
			i = k;
			// Maximum over lower diagonal elements per column
			ta = abs(pv[kpos + k]);
			for (j = k + 1; j < n; j++)
				if ((tb = abs(pv[j*n + k])) > ta)
				{
					ta = tb;
					i = j;
				}

			//if (ta < epsilon(a))
			//	return false;

			// Swapping rows to put max over column to diagonal
			if (i != k)
			{
				std::swap(ri[k], ri[i]);
				for (ipos = i * n, j = 0; j < n; j++)
					std::swap(pv[kpos + j], pv[ipos + j]);
			}

			// Divide by max
			a = T(1) / pv[kpos + k];
			pv[kpos + k] = T(1);

			for (j = 0; j < n; j++)
				pv[kpos + j] *= a;

			// Elimination
			for (i = 0; i < n; i++)
			{
				if (i != k)
				{
					ipos = i * n;
					a = pv[ipos + k];
					pv[ipos + k] = T(0);
					for (j = 0; j < n; j++)
						pv[ipos + j] -= a * pv[kpos + j];
				}
			}
		}
		for (j = 0; j < n; j++)
		{
			if (j != ri[j])         // Column is out of order
			{
				k = j + 1;
				while (j != ri[k])
					k++;
				for (i = 0; i < n; i++)
					std::swap(pv[i*n + j], pv[i*n + k]);
				std::swap(ri[j], ri[k]);
			}
		}
		return true;
	}
	template <typename T>
	bool Matrix<T>::lu(std::valarray<size_t>& ri, T* pDet)
	{
		assert(M == N);
		size_t i, j, k;
		double ta, tb;

		if (M != ri.size())
			ri.resize(M);

		if (pDet != NULL)
			*pDet = T(1);

		size_t n = M;
		T *pv = &this->values[0];

		for (i = 0; i < n; i++)
			ri[i] = i;

		for (k = 0; k < n - 1; k++)
		{
			j = k;
			ta = abs(pv[ri[k] * n + k]);
			for (i = k + 1; i < n; i++)
				if ((tb = abs(pv[ri[i] * n + k])) > ta)
				{
					ta = tb;
					j = i;
				}
			if (j != k)
			{
				std::swap(ri[j], ri[k]);
				if (pDet != NULL)
					*pDet = -*pDet;
			}
			size_t kpos = ri[k] * n;

			if (abs(pv[kpos + k]) < epsilon(pv[kpos + k]))
				return false;

			if (pDet != NULL)
				*pDet *= pv[kpos + k];

			for (i = k + 1; i < n; i++)
			{
				size_t ipos = ri[i] * n;
				T a = pv[ipos + k] /= pv[kpos + k];

				for (j = k + 1; j < n; j++)
					pv[ipos + j] -= a * pv[kpos + j];
			}
		}
		if (pDet != NULL)
			*pDet *= pv[ri[k] * n + k];

		return true;
	}
	template <typename T> 
	T Matrix<T>::det() const
	{
		T d;

		Matrix<T> m(*this);
		std::valarray<size_t> ri(M);
		if (!m.lu(ri, &d))
			d = T(0);

		return d;
	}
	template <class T> inline
	T sign(T a, T b)
	{
		return (b >= T(0) ? abs(a) : -abs(a));
	}
	template <typename T> 
	bool Matrix<T>::svd(Matrix<T>& vc, std::valarray<T>& w)
	{
		size_t flag, i, its, j, jj, k, l, nm;
		T c, f, h, s, x, y, z, tmp;
		size_t m = this->M;
		size_t n = this->N;

		/*if (vc.M != n || vc.N != n)
			vc = Matrix<T>(n, n);
		if (w.size() != n)
			w.resize(n);*/

		T *a = &values[0];
		T *v = &vc.values[0];
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
					scale += abs(a[k*n + i]);

				if (scale > epsilon(scale))
				{
					for (k = i; k < m; k++)
					{
						tmp = a[k*n + i] /= scale;
						s += tmp * tmp;
					}
					f = a[i*n + i];
					g = -sign(sqrt(s), f);
					h = f * g - s;
					a[i*n + i] = f - g;

					for (j = l; j < n; j++)
					{
						for (s = T(0), k = i; k < m; k++)
							s += a[k*n + i] * a[k*n + j];
						f = s / h;
						for (k = i; k < m; k++)
							a[k*n + j] += f * a[k*n + i];
					}
					for (k = i; k < m; k++)
						a[k*n + i] *= scale;
				}
			}
			w[i] = scale * g;
			g = s = scale = T(0);
			if (i < m && i != n - 1)
			{
				for (k = l; k < n; k++)
					scale += abs(a[i*n + k]);

				if (scale > epsilon(scale))
				{
					for (k = l; k < n; k++)
					{
						tmp = a[i*n + k] /= scale;
						s += tmp * tmp;
					}
					f = a[i*n + l];
					g = -sign(sqrt(s), f);
					h = f * g - s;
					a[i*n + l] = f - g;

					for (k = l; k < n; k++)
						rv1[k] = a[i*n + k] / h;

					for (j = l; j < m; j++)
					{
						for (s = T(0), k = l; k < n; k++)
							s += a[j*n + k] * a[i*n + k];
						for (k = l; k < n; k++)
							a[j*n + k] += s * rv1[k];
					}
					for (k = l; k < n; k++)
						a[i*n + k] *= scale;
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
						v[j*n + i] = (a[i*n + j] / a[i*n + l]) / g;
					for (j = l; j < n; j++)
					{
						for (s = T(0), k = l; k < n; k++)
							s += a[i*n + k] * v[k*n + j];

						for (k = l; k < n; k++)
							v[k*n + j] += s * v[k*n + i];
					}
				}
				for (j = l; j < n; j++)
					v[i*n + j] = v[j*n + i] = T(0);
			}
			v[i*n + i] = T(1);
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
				a[i*n + j] = T(0);
			if (abs(g) > epsilon(g))
			{
				g = T(1) / g;
				for (j = l; j < n; j++)
				{
					for (s = T(0), k = l; k < m; k++)
						s += a[k*n + i] * a[k*n + j];

					f = (s / a[i*n + i]) * g;

					for (k = i; k < m; k++)
						a[k*n + j] += f * a[k*n + i];
				}
				for (j = i; j < m; j++)
					a[j*n + i] *= g;

			}
			else
			{
				for (j = i; j < m; j++)
					a[j*n + i] = T(0);
			}
			++a[i*n + i];
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
							y = a[j*n + nm];
							z = a[j*n + i];
							a[j*n + nm] = y * c + z * s;
							a[j*n + i] = z * c - y * s;
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
							v[j*n + k] = -v[j*n + k];
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
						x = v[jj*n + j];
						z = v[jj*n + i];
						v[jj*n + j] = x * c + z * s;
						v[jj*n + i] = z * c - x * s;
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
						y = a[jj*n + j];
						z = a[jj*n + i];
						a[jj*n + j] = y * c + z * s;
						a[jj*n + i] = z * c - y * s;
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

	
	template <class T> inline
	void cdiv(const T& xr, const T& xi, const T& yr, const T& yi, T& cdivr, T& cdivi)
	{
		T r, d;

		if (abs(yr) > abs(yi))
		{
			r = yi / yr;
			d = yr + r * yi;
			cdivr = (xr + r * xi) / d;
			cdivi = (xi - r * xr) / d;
		}
		else
		{
			r = yr / yi;
			d = yi + r * yr;
			cdivr = (r*xr + xi) / d;
			cdivi = (r*xi - xr) / d;
		}
	}
	template <typename T>
	void Matrix<T>::balanc(Matrix<T>& v, bool eivec)
	{
		size_t i, j, lo, hi, m, n;


		n = M;
		lo = 0;
		hi = n - 1;
		std::valarray<T> ort(n);
		T *pv = nullptr, *pm = &v.values[0];
		if (eivec)
			pv = &v(0, 0);

		for (m = lo + 1; m <= hi - 1; m++)
		{
			T scale(0);
			for (i = m; i <= hi; i++)
				scale += abs(pm[i*n + m - 1]);

			if (scale > epsilon(scale))
			{
				T h(0);
				for (i = hi; i >= m; i--)
				{
					ort[i] = pm[i*n + m - 1] / scale;
					h += ort[i] * ort[i];
				}
				T g = sqrt(h);
				if (ort[m] > T(0))
					g = -g;

				h -= ort[m] * g;
				ort[m] -= g;

				for (j = m; j < n; j++)
				{
					T f(0);
					for (i = hi; i >= m; i--)
						f += ort[i] * pm[i*n + j];
					f /= h;
					for (i = m; i <= hi; i++)
						pm[i*n + j] -= f * ort[i];
				}

				for (i = 0; i <= hi; i++)
				{
					T f(0);
					for (j = hi; j >= m; j--)
						f += ort[j] * pm[i*n + j];
					f /= h;
					for (j = m; j <= hi; j++)
						pm[i*n + j] -= f * ort[j];
				}
				ort[m] = scale * ort[m];
				pm[m*n + m - 1] = scale * g;
			}
		}

		if (eivec)
			for (i = 0; i < n; i++)
				for (j = 0; j < n; j++)
					pv[i*n + j] = (i == j ? T(1) : T(0));

		for (m = hi - 1; m >= lo + 1; m--)
		{
			if (abs(pm[m*n + m - 1]) > T(0))
			{
				for (i = m + 1; i <= hi; i++)
					ort[i] = pm[i*n + m - 1];

				if (eivec)
					for (j = m; j <= hi; j++)
					{
						T g(0);
						for (i = m; i <= hi; i++)
							g += ort[i] * pv[i*n + j];

						g = (g / ort[m]) / pm[m*n + m - 1];
						for (i = m; i <= hi; i++)
							pv[i*n + j] += g * ort[i];
					}
			}
		}
	}
	template <typename  T> 
	bool Matrix<T>::hqr2(std::valarray<T>& d, std::valarray<T>& e, Matrix<T>& v, bool eivec)
	{
	   int i,j,k,l;

	   int nn = this->N;
	   int n = nn-1;
	   int low = 0;
	   int high = nn-1;
	   T exshift(0);
	   T p(0),q(0),r(0),s(0),z(0),t,w,x,y;
	   T cdivr, cdivi;

	   T *pv = nullptr, *pm = &this->values[0];
	   if (eivec)
		  pv = &v(0,0);

	   T norm(0);
	   for (i=0; i < nn; i++) 
	   {
		  if (i < low || i > high) 
		  {
			 d[i] = pm[i*nn+i];
			 e[i] = T(0);
		  }
		  for (j = std::max(i-1,0); j < nn; j++)
			 norm += abs( pm[i*nn+j]);
	   }

	   size_t iter = 0;
	   while (n >= low) 
	   {
		  l = n;
		  while (l > low) 
		  {
			 s = abs( pm[(l-1)*nn+l-1]) + abs( pm[l*nn+l]);
			 if (s < epsilon(s))
				s = norm;
        
			 if (abs( pm[l*nn+l-1]) < epsilon(pm[l*nn+l-1]) * s)
				break;
        
			 l--;     
		  }
    
		  if (l == n) 
		  {
			 pm[n*nn+n] += exshift;
			 d[n] = pm[n*nn+n];
			 e[n] = T(0);
			 n--;
			 iter = 0;
		  } 
		  else if (l == n-1) 
		  {
			 w = pm[n*nn+n-1] * pm[(n-1)*nn+n];
			 p = (pm[(n-1)*nn+n-1] - pm[n*nn+n]) / T(2);
			 q = p * p + w;
			 z = sqrt( abs( q));
			 pm[n*nn+n] += exshift;
			 pm[(n-1)*nn+n-1] += exshift;
			 x = pm[n*nn+n];

			 if (q >= T(0)) 
			 {
				if (p >= T(0))
				   z = p + z;
				else
				   z = p - z;
           
				d[n-1] = x + z;
				d[n] = d[n-1];
				if (z != T(0))
				   d[n] = x - w / z;
           
				e[n-1] = T(0);
				e[n] = T(0);

				x = pm[n*nn+n-1];
				s = abs( x) + abs( z);
				p = x / s;
				q = z / s;
				r = sqrt( p*p + q*q);
				p = p / r;
				q = q / r;

				for (j = n-1; j < nn; j++) 
				{
				   z = pm[(n-1)*nn+j];
				   pm[(n-1)*nn+j] = q * z + p * pm[n*nn+j];
				   pm[n*nn+j] = q * pm[n*nn+j] - p * z;
				}

				for (i = 0; i <= n; i++) 
				{
				   z = pm[i*nn+n-1];
				   pm[i*nn+n-1] = q * z + p * pm[i*nn+n];
				   pm[i*nn+n] = q * pm[i*nn+n] - p * z;
				}

				if (eivec)
				   for (i = low; i <= high; i++) 
				   {
					  z = pv[i*nn+n-1];
					  pv[i*nn+n-1] = q * z + p * pv[i*nn+n];
					  pv[i*nn+n] = q * pv[i*nn+n] - p * z;
				   }
			 } 
			 else 
			 {
				d[n-1] = x + p;
				d[n] = x + p;
				e[n-1] = z;
				e[n] = -z;
			 }
			 n = n - 2;
			 iter = 0;
		  } 
		  else 
		  {
			 x = pm[n*nn+n];
			 y = T(0);
			 w = T(0);
			 if (l < n) 
			 {
				y = pm[(n-1)*nn+n-1];
				w = pm[n*nn+n-1] * pm[(n-1)*nn+n];
			 }

			 if (iter == 10) 
			 {
				exshift += x;
				for (i = low; i <= n; i++)
				   pm[i*nn+i] -= x;
           
				s = abs( pm[n*nn+n-1]) + abs( pm[(n-1)*nn+n-2]);
				x = y = 0.75 * s;
				w = -0.4375 * s * s;
			 }

			 if (iter == 30) 
			 {
				 s = (y - x) / T(2);
				 s = s * s + w;
				 if (s > T(0)) 
				 {
					 s = sqrt( s);
					 if (y < x)
						s = -s;
                
					 s = x - w / ((y - x) / T(2) + s);
					 for (i = low; i <= n; i++)
						pm[i*nn+i] -= s;
                 
					 exshift += s;
					 x = y = w = 0.964;
				 }
			 }

			 if (++iter > 250)
				return false;

			 int m = n-2;
			 while (m >= l) 
			 {
				z = pm[m*nn+m];
				r = x - z;
				s = y - z;
				p = (r * s - w) / pm[(m+1)*nn+m] + pm[m*nn+m+1];
				q = pm[(m+1)*nn+m+1] - z - r - s;
				r = pm[(m+2)*nn+m+1];
				s = abs( p) + abs( q) + abs( r);
				p = p / s;
				q = q / s;
				r = r / s;
				if (m == l)
				   break;
            
				if (abs( pm[m*nn+m-1]) * (abs( q) + abs( r)) <
				   epsilon(r) * (abs( p) * (abs( pm[(m-1)*nn+m-1]) + abs( z) +
				   abs( pm[(m+1)*nn+m+1])))) {
					  break;
				}
				m--;
			 }

			 for (i = m+2; i <= n; i++) 
			 {
				pm[i*nn+i-2] = T(0);
				if (i > m+2)
				   pm[i*nn+i-3] = T(0);
			 }

			 for (k = m; k <= n-1; k++) 
			 {
				bool notlast = (k != n-1);
				if (k != m) 
				{
				   p = pm[k*nn+k-1];
				   q = pm[(k+1)*nn+k-1];
				   r = notlast ? pm[(k+2)*nn+k-1] : T(0);
				   x = abs( p) + abs( q) + abs( r);
				   if (x > epsilon(x)) 
				   {
					  p = p / x;
					  q = q / x;
					  r = r / x;
				   }
				}
				if (x < epsilon(x))
				   break;
            
				s = sqrt( p * p + q * q + r * r);
				if (p < T(0))
				   s = -s;
            
				if (s != T(0)) 
				{
				   if (k != m)
					  pm[k*nn+k-1] = -s * x;
				   else if (l != m)
					  pm[k*nn+k-1] = -pm[k*nn+k-1];
               
				   p = p + s;
				   x = p / s;
				   y = q / s;
				   z = r / s;
				   q = q / p;
				   r = r / p;

				   for (j = k; j < nn; j++) 
				   {
					  p = pm[k*nn+j] + q * pm[(k+1)*nn+j];
					  if (notlast) 
					  {
						 p = p + r * pm[(k+2)*nn+j];
						 pm[(k+2)*nn+j] = pm[(k+2)*nn+j] - p * z;
					  }
					  pm[k*nn+j] = pm[k*nn+j] - p * x;
					  pm[(k+1)*nn+j] = pm[(k+1)*nn+j] - p * y;
				   }

				   for (i = 0; i <= std::min( n, k+3); i++) 
				   {
					  p = x * pm[i*nn+k] + y * pm[i*nn+k+1];
					  if (notlast) 
					  {
						 p = p + z * pm[i*nn+k+2];
						 pm[i*nn+k+2] = pm[i*nn+k+2] - p * r;
					  }
					  pm[i*nn+k] = pm[i*nn+k] - p;
					  pm[i*nn+k+1] = pm[i*nn+k+1] - p * q;
				   }

				   if (eivec)
					  for (i = low; i <= high; i++) 
					  {
						 p = x * pv[i*nn+k] + y * pv[i*nn+k+1];
						 if (notlast) 
						 {
							p += z * pv[i*nn+k+2];
							pv[i*nn+k+2] -= p * r;
						 }
						 pv[i*nn+k] -= p;
						 pv[i*nn+k+1] -= p * q;
					  }
				}
			 }
		  }
	   }

	   if (norm < epsilon(norm))
		  return true;

	   for (n = nn-1; n >= 0; n--) 
	   {
		  p = d[n];
		  q = e[n];

		  if (q == T(0)) 
		  {
			 int l = n;
			 pm[n*nn+n] = T(1);
			 for (i = n-1; i >= 0; i--) 
			 {
				w = pm[i*nn+i] - p;
				r = T(0);
				for (j = l; j <= n; j++)
				   r += pm[i*nn+j] * pm[j*nn+n];
            
				if (e[i] < T(0)) 
				{
				   z = w;
				   s = r;
				} 
				else 
				{
				   l = i;
				   if (e[i] == T(0)) 
				   {
					  if (w != T(0))
						 pm[i*nn+n] = -r / w;
					  else
						 pm[i*nn+n] = -r / (epsilon(norm) * norm);
				   } 
				   else 
				   {
					  x = pm[i*nn+i+1];
					  y = pm[(i+1)*nn+i];
					  q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
					  t = (x * s - z * r) / q;
					  pm[i*nn+n] = t;
					  if (abs( x) > abs( z))
						 pm[(i+1)*nn+n] = (-r - w * t) / x;
					  else
						 pm[(i+1)*nn+n] = (-s - y * t) / z;
				   }

				   t = abs( pm[i*nn+n]);
				   if ((epsilon(t) * t) * t > T(1)) 
					  for (j = i; j <= n; j++)
						 pm[j*nn+n] /= t;
				}
			 }
		  } 
		  else if (q < T(0)) 
		  {
			 int l = n-1;

			 if (abs( pm[n*nn+n-1]) > abs( pm[(n-1)*nn+n])) 
			 {
				pm[(n-1)*nn+n-1] = q / pm[n*nn+n-1];
				pm[(n-1)*nn+n] = -(pm[n*nn+n] - p) / pm[n*nn+n-1];
			 } 
			 else 
			 {
				cdiv( 0.0, -pm[(n-1)*nn+n], pm[(n-1)*nn+n-1]-p, q, cdivr, cdivi);
				pm[(n-1)*nn+n-1] = cdivr;
				pm[(n-1)*nn+n] = cdivi;
			 }
			 pm[n*nn+n-1] = T(0);
			 pm[n*nn+n] = T(1);
			 for (i = n-2; i >= 0; i--) 
			 {
				T ra(0),sa(0),vr,vi;

				for (j = l; j <= n; j++) 
				{
				   ra += pm[i*nn+j] * pm[j*nn+n-1];
				   sa += pm[i*nn+j] * pm[j*nn+n];
				}
				w = pm[i*nn+i] - p;

				if (e[i] < T(0)) 
				{
				   z = w;
				   r = ra;
				   s = sa;
				} 
				else 
				{
				   l = i;
				   if (e[i] == T(0))
				   {
					  cdiv( -ra, -sa, w, q, pm[i*nn+n-1], pm[i*nn+n]);
				   } 
				   else 
				   {
					  x = pm[i*nn+i+1];
					  y = pm[(i+1)*nn+i];
					  vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
					  vi = (d[i] - p) * 2.0 * q;
					  if (vr == T(0) && vi == T(0)) 
						 vr = epsilon(norm) * norm * (abs( w) + abs(q) + abs(x) + abs(y) + abs(z));

					  cdiv( x*r - z*ra + q*sa, x*s - z*sa - q*ra, vr, vi, cdivr, cdivi);
					  pm[i*nn+n-1] = cdivr;
					  pm[i*nn+n] = cdivi;
					  if (abs(x) > (abs(z) + abs(q))) 
					  {
						 pm[(i+1)*nn+n-1] = (-ra - w * pm[i*nn+n-1] + q * pm[i*nn+n]) / x;
						 pm[(i+1)*nn+n] = (-sa - w * pm[i*nn+n] - q * pm[i*nn+n-1]) / x;
					  } 
					  else 
					  {
						 cdiv( -r - y * pm[i*nn+n-1], -s - y * pm[i*nn+n], z, q, cdivr, cdivi);
						 pm[(i+1)*nn+n-1] = cdivr;
						 pm[(i+1)*nn+n] = cdivi;
					  }
				   }

				   t = std::max( abs( pm[i*nn+n-1]), abs( pm[i*nn+n]));
				   if ((epsilon(t) * t) * t > T(1)) 
					  for (j = i; j <= n; j++) 
					  {
						 pm[j*nn+n-1] /= t;
						 pm[j*nn+n] /= t;
					  }
				}
			 }
		  }
	   }

	   if (eivec)
	   {
		  for (i = 0; i < nn; i++) 
			 if (i < low || i > high) 
				for (j = i; j < nn; j++)
				   pv[i*nn+j] = pm[i*nn+j];
         
		  for (j = nn-1; j >= low; j--) 
			 for (i = low; i <= high; i++) 
			 {
				z = T(0);
				for (k = low; k <= std::min( j, high); k++)
				   z += pv[i*nn+k] * pm[k*nn+j];
         
				pv[i*nn+j] = z;
			 }
	   }
	   return true;
	}
	template <typename T> 
	bool Matrix<T>::eigen(std::valarray<T>& rev, std::valarray<T>& iev) const
	{
		assert(this->M == this->N);

		if (rev.size() != this->M)
			rev.resize(M);
		if (iev.size() != this->M)
			iev.resize(this->M);

		Matrix<T> eivec(this->M, this->N), m(*this);
		m.balanc(eivec, false);

		return m.hqr2(rev, iev, eivec, false);
	}
}

#endif /* MATRIX_HPP */
