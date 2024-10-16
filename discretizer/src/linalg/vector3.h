#ifndef CPP_VECTOR3_H
#define CPP_VECTOR3_H

#include <cmath>
#include <array>
#include "linalg/matrix.h"

namespace linalg
{
	const uint8_t ND = 3;
	class Vector3
	{
	public:

		Vector3()
		{
			this->x = this->y = this->z = 0.0;
		};
		Vector3(value_t x_, value_t y_, value_t z_)
		{
			this->x = x_;
			this->y = y_;
			this->z = z_;
		};
		Vector3(const Vector3& vec)
		{
			(*this) = vec;
		};
		Vector3& operator=(const Vector3& m)
		{
			this->x = m.x;
			this->y = m.y;
			this->z = m.z;
			return *this;
		};

		union
		{
			std::array<value_t, ND> values;
			struct
			{
				value_t x;
				value_t y;
				value_t z;
			};
		};
	public:

		// incrementation operators
		Vector3& operator+=(const Vector3& v)
		{
			this->x += v.x;
			this->y += v.y;
			this->z += v.z;
			return *this;
		};
		Vector3& operator-=(const Vector3& v)
		{
			this->x -= v.x;
			this->y -= v.y;
			this->z -= v.z;
			return *this;
		};
		Vector3& operator*=(const value_t k)
		{
			this->x *= k;
			this->y *= k;
			this->z *= k;
			return *this;
		};
		Vector3& operator/=(const value_t k)
		{
			this->x /= k;
			this->y /= k;
			this->z /= k;
			return *this;
		};
		[[nodiscard]] value_t norm() const
		{
			return sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
		}
		[[nodiscard]] value_t norm_sq() const
		{
			return this->x * this->x + this->y * this->y + this->z * this->z;
		}
	};

	inline Vector3 operator-(const Vector3& v)
	{
		return {-v.x, -v.y, -v.z};
	}
	inline std::ostream& operator<<(std::ostream& out, const Vector3& v)
	{
		return out << v.x << ' ' << v.y << ' ' << v.z;
	}

	inline Vector3 operator+(const Vector3& u, const Vector3& v)
	{
		return { u.x + v.x, u.y + v.y, u.z + v.z };
	}

	inline Vector3 operator-(const Vector3& u, const Vector3& v)
	{
		return { u.x - v.x, u.y - v.y, u.z - v.z };
	}

	inline Vector3 operator*(const Vector3& u, const Vector3& v)
	{
		return { u.x * v.x, u.y * v.y, u.z * v.z };
	}
	inline Vector3 operator*(const Vector3& v, const value_t t)
	{
		return { t * v.x, t * v.y, t * v.z };
	}
	inline Vector3 operator*(const value_t t, const Vector3& v)
	{
		return { t * v.x, t * v.y, t * v.z };
	}
	inline Vector3 operator/(const Vector3& v, const value_t t)
	{
		return (1.0 / t) * v;
	}
	inline Vector3 cross(const Vector3& u, const Vector3& v)
	{
		return { u.y * v.z - u.z * v.y,
				u.z * v.x - u.x * v.z,
				u.x * v.y - u.y * v.x };
	}
	inline double dot(const Vector3& u, const Vector3& v)
	{
		return u.x * v.x + u.y * v.y + u.z * v.z;
	}
	inline double dot(const Matrix<value_t>& u, const Vector3& v)
	{
		assert((u.N == 1 && u.M == ND) || (u.N == ND && u.M == 1));
		return u.values[0] * v.x + u.values[1] * v.y + u.values[2] * v.z;
	}
	inline double dot(const Vector3& v, const Matrix<value_t>& u)
	{
		return dot(u, v);
	}
	inline Vector3 matrix_vector_product(const Matrix<value_t>& mat, const Vector3& vec)
	{
		Vector3 result;

		result.x = vec.x * mat(0, 0) + vec.y * mat(0, 1) + vec.z * mat(0, 2);
		result.y = vec.x * mat(1, 0) + vec.y * mat(1, 1) + vec.z * mat(1, 2);
		result.z = vec.x * mat(2, 0) + vec.y * mat(2, 1) + vec.z * mat(2, 2);

		return result;
	}
}

#endif /* CPP_VECTOR3_H */
