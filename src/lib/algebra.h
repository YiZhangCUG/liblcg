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
#include "config.h"

#ifdef LCG_FABS
/**
 * @brief      return absolute value
 *
 * @param      x     input value
 */
#define lcg_fabs(x) ((x < 0) ? -1*x : x)
#endif

/**
 * @brief      return the bigger value
 *
 * @param      a     input value
 * @param      b     another input value
 *
 * @return     the bigger value
 */
#define lcg_max(a, b) (a>b?a:b)

/**
 * @brief      return the smaller value
 *
 * @param      a     input value
 * @param      b     another input value
 *
 * @return     the smaller value
 */
#define lcg_min(a, b) (a<b?a:b)

/**
 * @brief      Set the input value within a box constraint
 *
 * @param      a     low boundary
 * @param      b     high boundary
 * @param      in    input value
 *
 * @return     box constrained value
 */
#define lcg_set2box(a, b, in) (lcg_max(a, lcg_min(b, in)))

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
 * @brief      Locate memory for a lcg_float pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
lcg_float* malloc(const int n);

/**
 * @brief      Destroy memory used by the lcg_float type array.
 *
 * @param      x     Pointer of the array.
 */
void free(lcg_float* x);

/**
 * @brief      calculate dot product of two real vectors
 *
 * @param[in]  a       pointer of the vector a
 * @param[in]  b       pointer of the vector b
 * @param[in]  size    size of the vector
 *
 * @return     dot product
 */
lcg_float dot(const lcg_float *a, const lcg_float *b, int size);

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
void matvec(lcg_float **A, const lcg_float *x, lcg_float *Ax, int m_size, int n_size, 
	matrix_layout_e layout = Normal);

/**
 * @brief      Calculate the sum of two vectors
 *
 * @param[in]  a     vector a
 * @param[in]  b     vector b
 * @param      sum   vector sum
 * @param[in]  size  vector size
 */
void addvec(const lcg_float *a, const lcg_float *b, lcg_float *sum, int size);

/**
 * @brief      Calculate the difference of two vectors
 *
 * @param[in]  a     vector a
 * @param[in]  b     vector b
 * @param      sub   vector sub
 * @param[in]  size  vector size
 */
void subvec(const lcg_float *a, const lcg_float *b, lcg_float *sub, int size);

/**
 * @brief      Append a vector to another
 *
 * @param[in]  a      vector a
 * @param      ret    vector ret
 * @param[in]  size   vector size
 * @param[in]  scale  scale
 */
void appvec(const lcg_float *a, lcg_float *ret, int size, lcg_float scale = 1.0);

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
 * @brief      Locate memory for a lcg_complex pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
lcg_complex* malloc_complex(const int n);

/**
 * @brief      Destroy memory used by the lcg_complex type array.
 *
 * @param      x     Pointer of the array.
 */
void free(lcg_complex* x);

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
lcg_complex dot_complex(const lcg_complex *a, const lcg_complex *b, int x_size);

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
lcg_complex inner_complex(const lcg_complex *a, const lcg_complex *b, int x_size);

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
void matvec_complex(lcg_complex **A, const lcg_complex *x, lcg_complex *Ax, int m_size, int n_size, 
	matrix_layout_e layout = Normal, complex_conjugate_e conjugate = NonConjugate);

#endif //_ALGEBRA_H