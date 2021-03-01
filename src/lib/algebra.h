/******************************************************//**
 *    C++ library of real and complex linear algebra.
 *
 * Copyright (c) 2019-2029 Yi Zhang (zhangyiss@icloud.com)
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *********************************************************/

#ifndef _ALGEBRA_H
#define _ALGEBRA_H

#include "iostream"


/**
 * @brief      Matrix layouts.
 */
enum matrix_layout_e
{
	Normal,
	Transpose,
};

/**
 * @brief      Conjugate types for a complex number.
 */
enum complex_conjugate_e
{
	NonConjugate,
	Conjugate,
};

/**
 * @brief      A simple definition of the float type we use here. 
 * Easy to change in the future. Right now it is just an alias of double
 */
typedef double lcg_float;

/**
 * @brief     A simple definition of the complex number type. 
 * Easy to change in the future. Right now it is just two double variables
 */
struct lcg_complex
{
	lcg_float rel; ///< The real part
	lcg_float img; ///< The imaginary part

	/**
	 * @brief      Constructs a new instance.
	 */
	lcg_complex();
	/**
	 * @brief      Constructs a new instance.
	 *
	 * @param[in]  r     The real part of the complex number
	 * @param[in]  i     The imaginary part of the complex number
	 */
	lcg_complex(lcg_float r, lcg_float i);
	/**
	 * @brief      Destructor
	 */
	virtual ~lcg_complex();
	/**
	 * @brief      setup a complex number
	 *
	 * @param[in]  r     The real part of the complex number
	 * @param[in]  i     The imaginary part of the complex number
	 */
	void set(lcg_float r, lcg_float i);
	/**
	 * @brief      Calculate the L2 module of a complex number
	 *
	 * @return     The module
	 */
	lcg_float module();
	/**
	 * @brief      Calculate the conjugate of a complex number
	 *
	 * @return     The complex conjugate.
	 */
	lcg_complex conjugate();
};

/**
 * @brief      Reload equality operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     equal or not
 */
bool operator==(const lcg_complex &a, const lcg_complex &b);

/**
 * @brief      Reload inequality operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     unequal or not
 */
bool operator!=(const lcg_complex &a, const lcg_complex &b);

/**
 * @brief      Reload addition operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     sum
 */
lcg_complex operator+(const lcg_complex &a, const lcg_complex &b);

/**
 * @brief      Reload subtraction operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     subtraction
 */
lcg_complex operator-(const lcg_complex &a, const lcg_complex &b);

/**
 * @brief      Reload multiplication operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     product
 */
lcg_complex operator*(const lcg_complex &a, const lcg_complex &b);

/**
 * @brief      Reload multiplication operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     product
 */
lcg_complex operator*(const lcg_float &a, const lcg_complex &b);

/**
 * @brief      Reload division operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     quotient
 */
lcg_complex operator/(const lcg_complex &a, const lcg_complex &b);

/**
 * @brief      Reload ostream operator.
 *
 * @param      os    The ostream
 * @param[in]  a     complex number a
 *
 * @return     The ostream
 */
std::ostream &operator<<(std::ostream &os, const lcg_complex &a);

/**
 * @brief      Return absolute value
 *
 * @param[in]  a     input value
 *
 * @return     The absolute value
 */
inline static lcg_float lcg_abs(lcg_float a)
{
	if (a >= 0.0) return a;
	return -1.0*a;
}

/**
 * @brief      Return the bigger value
 *
 * @param[in]  a     input value
 * @param[in]  b     input value
 *
 * @return     The bigger value
 */
inline static lcg_float lcg_max(lcg_float a, lcg_float b)
{
	if (a >= b) return a;
	return b;
}

/**
 * @brief      Return the smaller value
 *
 * @param[in]  a     input value
 * @param[in]  b     input value
 *
 * @return     The smaller value
 */
inline static lcg_float lcg_min(lcg_float a, lcg_float b)
{
	if (a <= b) return a;
	return b;
}

/**
 * @brief      Set the input value within a box constraint
 *
 * @param      a     low boundary
 * @param      b     high boundary
 * @param      in    input value
 * @param      low_bound    Whether to include the low boundary value
 * @param      hig_bound    Whether to include the high boundary value
 *
 * @return     box constrained value
 */
inline static lcg_float lcg_set2box(lcg_float low, lcg_float hig, lcg_float a, 
	bool low_bound = true, bool hig_bound = true)
{
	if (hig_bound && a >= hig) return hig;
	if (!hig_bound && a >= hig) return (hig - 1e-16);
	if (low_bound && a <= low) return low;
	if (!low_bound && a <= low) return (low + 1e-16);
	return a;
}

/**
 * @brief      Locate memory for a lcg_float pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
inline static lcg_float* lcg_malloc(int n)
{
	lcg_float* x = new lcg_float [n];
	return x;
}

/**
 * @brief      Locate memory for a lcg_float second pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
inline static lcg_float** lcg_malloc(int m, int n)
{
	lcg_float **x = new lcg_float* [m];
	for (int i = 0; i < m; i++)
	{
		x[i] = new lcg_float [n];
	}
	return x;
}

/**
 * @brief      Locate memory for a lcg_complex pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
inline static lcg_complex* lcg_malloc_complex(int n)
{
	lcg_complex* x = new lcg_complex [n];
	return x;
}

/**
 * @brief      Locate memory for a lcg_complex second pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
inline static lcg_complex** lcg_malloc_complex(int m, int n)
{
	lcg_complex **x = new lcg_complex* [m];
	for (int i = 0; i < m; i++)
	{
		x[i] = new lcg_complex [n];
	}
	return x;
}

/**
 * @brief      Destroy memory used by the lcg_float type array.
 *
 * @param      x     Pointer of the array.
 */
inline static void lcg_free(lcg_float* x)
{
	if (x != nullptr)
	{
		delete[] x;
		x = nullptr;
	}
	return;
}

/**
 * @brief      Destroy memory used by the 2D lcg_float type array.
 *
 * @param      x     Pointer of the array.
 */
inline static void lcg_free(lcg_float **x, int m)
{
	if (x != nullptr)
	{
		for (int i = 0; i < m; i++)
		{
			delete[] x[i];
		}
		delete[] x;
		x = nullptr;
	}
	return;
}

/**
 * @brief      Destroy memory used by the lcg_complex type array.
 *
 * @param      x     Pointer of the array.
 */
inline static void lcg_free(lcg_complex* x)
{
	if (x != nullptr)
	{
		delete[] x;
		x = nullptr;
	}
	return;
}

/**
 * @brief      Destroy memory used by the 2D lcg_complex type array.
 *
 * @param      x     Pointer of the array.
 */
inline static void lcg_free(lcg_complex **x, int m)
{
	if (x != nullptr)
	{
		for (int i = 0; i < m; i++)
		{
			delete[] x[i];
		}
		delete[] x;
		x = nullptr;
	}
	return;
}

/**
 * @brief      set a vector's value
 *
 * @param      a     pointer of the vector
 * @param[in]  b     initial value
 * @param[in]  size  vector size
 */
inline static void lcg_vecset(lcg_float *a, lcg_float b, int size)
{
	for (int i = 0; i < size; i++)
	{
		a[i] = b;
	}
	return;
}

/**
 * @brief      set a complex vector's value
 *
 * @param      a     pointer of the vector
 * @param[in]  b     initial value
 * @param[in]  size  vector size
 */
inline static void lcg_vecset(lcg_complex *a, lcg_complex b, int size)
{
	for (int i = 0; i < size; i++)
	{
		a[i] = b;
	}
	return;
}

/**
 * @brief      calculate dot product of two real vectors
 *
 * @param[in]  a       pointer of the vector a
 * @param[in]  b       pointer of the vector b
 * @param[in]  size    size of the vector
 *
 * @return     dot product
 */
inline static void lcg_dot(lcg_float &ret, const lcg_float *a, 
	const lcg_float *b, int size)
{
	ret = 0.0;
	for (int i = 0; i < size; i++)
	{
		ret += a[i]*b[i];
	}
	return;
}

/**
 * @brief      calculate dot product of two complex vectors
 * 
 * the product of two complex vectors are defined as <a, b> = \sum{a_i \cdot b_i}
 *
 * @param[in]  a       complex vector a
 * @param[in]  b       complex vector b
 * @param[in]  x_size  size of the vector
 *
 * @return     product
 */
inline static void lcg_dot(lcg_complex &ret, const lcg_complex *a, 
	const lcg_complex *b, int size)
{
	ret.set(0.0, 0.0);
	// <a,b> = \sum{a_i \cdot b_i}
	for (int i = 0; i < size; i++)
	{
		ret.rel += (a[i].rel*b[i].rel - a[i].img*b[i].img);
		ret.img += (a[i].rel*b[i].img + a[i].img*b[i].rel);
	}
	return;
}

/**
 * @brief      calculate inner product of two complex vectors
 * 
 * the product of two complex vectors are defined as <a, b> = \sum{\bar{a_i} \cdot b_i}
 *
 * @param[in]  a       complex vector a
 * @param[in]  b       complex vector b
 * @param[in]  x_size  size of the vector
 *
 * @return     product
 */
inline static void lcg_inner(lcg_complex &ret, const lcg_complex *a, 
	const lcg_complex *b, int size)
{
	ret.set(0.0, 0.0);
	// <a,b> = \sum{\bar{a_i} \cdot b_i}
	for (int i = 0; i < size; i++)
	{
		ret.rel += (a[i].rel*b[i].rel + a[i].img*b[i].img);
		ret.img += (a[i].rel*b[i].img - a[i].img*b[i].rel);
	}
	return;
}

/**
 * @brief      calculate product of a real matrix and a vector
 * 
 * Different configurations:
 * layout=Normal -> A
 * layout=Transpose -> A^T
 *
 * @param      A          matrix A
 * @param[in]  x          vector x
 * @param      Ax         product of Ax
 * @param[in]  m_size     row size of A
 * @param[in]  n_size     column size of A
 * @param[in]  layout     layout of A used for multiplication. Must be Normal or Transpose
 */
void lcg_matvec(lcg_float **A, const lcg_float *x, lcg_float *Ax, int m_size, int n_size, 
	matrix_layout_e layout = Normal);

/**
 * @brief      calculate product of a complex matrix and a complex vector
 * 
 * the product of two complex vectors are defined as <a, b> = \sum{\bar{a_i}\cdot\b_i}.
 * Different configurations:
 * layout=Normal,conjugate=false -> A
 * layout=Transpose,conjugate=false -> A^T
 * layout=Normal,conjugate=true -> \bar{A}
 * layout=Transpose,conjugate=true -> A^H
 *
 * @param      A          complex matrix A
 * @param[in]  x          complex vector x
 * @param      Ax         product of Ax
 * @param[in]  m_size     row size of A
 * @param[in]  n_size     column size of A
 * @param[in]  layout     layout of A used for multiplication. Must be Normal or Transpose
 * @param[in]  conjugate  whether to use the complex conjugate of A for calculation
 */
void lcg_matvec(lcg_complex **A, const lcg_complex *x, lcg_complex *Ax, int m_size, int n_size, 
	matrix_layout_e layout = Normal, complex_conjugate_e conjugate = NonConjugate);

#endif //_ALGEBRA_H