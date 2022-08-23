/******************************************************
 * C++ Library of the Linear Conjugate Gradient Methods (LibLCG)
 * 
 * Copyright (C) 2022  Yi Zhang (yizhang-geo@zju.edu.cn)
 * 
 * LibLCG is distributed under a dual licensing scheme. You can
 * redistribute it and/or modify it under the terms of the GNU Lesser
 * General Public License (LGPL) as published by the Free Software Foundation,
 * either version 2 of the License, or (at your option) any later version. 
 * You should have received a copy of the GNU Lesser General Public 
 * License along with this program. If not, see <http://www.gnu.org/licenses/>. 
 * 
 * If the terms and conditions of the LGPL v.2. would prevent you from
 * using the LibLCG, please consider the option to obtain a commercial
 * license for a fee. These licenses are offered by the LibLCG developing 
 * team. As a rule, licenses are provided "as-is", unlimited in time for 
 * a one time fee. Please send corresponding requests to: yizhang-geo@zju.edu.cn. 
 * Please do not forget to include some description of your company and the 
 * realm of its activities. Also add information on how to contact you by 
 * electronic and paper mail.
 ******************************************************/

#ifndef _LCG_COMPLEX_H
#define _LCG_COMPLEX_H

#include "iostream"

#include "algebra.h"
#ifdef LibLCG_STD_COMPLEX

#include "complex"

typedef std::complex<lcg_float> lcg_complex;

#else

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
	 * @brief      Set real part of a complex number
	 * 
	 * @param a    Input value
	 */
	void real(lcg_float a);

	/**
	 * @brief     Set image part of a complex number
	 * 
	 * @param a   Input value
	 */
	void imag(lcg_float a);

	/**
	 * @brief    Get real part of a complex number
	 * 
	 * @return lcg_float Real component
	 */
	lcg_float real();

	/**
	 * @brief    Get image part of a complex number
	 * 
	 * @return lcg_float Image component
	 */
	lcg_float imag();
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
 * @param[in]  a     real number a
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
 * @brief      Reload division operator.
 *
 * @param[in]  a     real number a
 * @param[in]  b     complex number b
 *
 * @return     quotient
 */
lcg_complex operator/(const lcg_float &a, const lcg_complex &b);

/**
 * @brief      Reload ostream operator.
 *
 * @param      os    The ostream
 * @param[in]  a     complex number a
 *
 * @return     The ostream
 */
std::ostream &operator<<(std::ostream &os, const lcg_complex &a);

#endif // LibLCG_STD_COMPLEX

/**
 * @brief      Locate memory for a lcg_complex pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
lcg_complex* clcg_malloc(int n);

/**
 * @brief      Locate memory for a lcg_complex second pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
lcg_complex** clcg_malloc(int m, int n);

/**
 * @brief      Destroy memory used by the lcg_complex type array.
 *
 * @param      x     Pointer of the array.
 */
void clcg_free(lcg_complex* x);

/**
 * @brief      Destroy memory used by the 2D lcg_complex type array.
 *
 * @param      x     Pointer of the array.
 */
void clcg_free(lcg_complex **x, int m);

/**
 * @brief      set a complex vector's value
 *
 * @param      a     pointer of the vector
 * @param[in]  b     initial value
 * @param[in]  size  vector size
 */
void clcg_vecset(lcg_complex *a, lcg_complex b, int size);

/**
 * @brief      set a 2d complex vector's value
 *
 * @param      a     pointer of the matrix
 * @param[in]  b     initial value
 * @param[in]  m     row size of the matrix
 * @param[in]  n     column size of the matrix
 */
void clcg_vecset(lcg_complex **a, lcg_complex b, int m, int n);

/**
 * @brief      setup a complex number
 *
 * @param[in]  r     The real part of the complex number
 * @param[in]  i     The imaginary part of the complex number
 */
void clcg_set(lcg_complex *a, lcg_float r, lcg_float i);

/**
 * @brief      Calculate the squared module of a complex number
 *
 * @return     The module
 */
lcg_float clcg_square(const lcg_complex *a);
/**
 * @brief      Calculate the module of a complex number
 *
 * @return     The module
 */
lcg_float clcg_module(const lcg_complex *a);
/**
 * @brief      Calculate the conjugate of a complex number
 *
 * @return     The complex conjugate.
 */
lcg_complex clcg_conjugate(const lcg_complex *a);

/**
 * @brief      set a complex vector using random values
 *
 * @param      a     pointer of the vector
 * @param[in]  l     the lower bound of random values
 * @param[in]  h     the higher bound of random values
 * @param[in]  size  size of the vector
 */
void clcg_vecrnd(lcg_complex *a, lcg_complex l, lcg_complex h, int size);

/**
 * @brief      set a 2D complex vector using random values
 *
 * @param      a     pointer of the vector
 * @param[in]  l     the lower bound of random values
 * @param[in]  h     the higher bound of random values
 * @param[in]  m     row size of the vector
 * @param[in]  n     column size of the vector
 */
void clcg_vecrnd(lcg_complex **a, lcg_complex l, lcg_complex h, int m, int n);

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
void clcg_dot(lcg_complex &ret, const lcg_complex *a, const lcg_complex *b, int size);

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
void clcg_inner(lcg_complex &ret, const lcg_complex *a, const lcg_complex *b, int size);

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
void clcg_matvec(lcg_complex **A, const lcg_complex *x, lcg_complex *Ax, int m_size, int n_size, 
	lcg_matrix_e layout = MatNormal, clcg_complex_e conjugate = NonConjugate);

#endif // _LCG_COMPLEX_H