/******************************************************//**
 *    C/C++ library of complex linear conjugate gradient.
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

#ifndef _CLCG_H
#define _CLCG_H

#ifdef __cplusplus
extern "C"
{

#include "lcg.h"
#endif

/**
 * @brief     A simple definition of the complex number type. 
 * Easy to change in the future. Right now it is just two lcg_float variables
 */
typedef struct
{
	lcg_float rel, img;
} clcg_complex;

/**
 * @brief      Matrix layouts.
 */
typedef enum
{
	Normal,
	Transpose,
} matrix_layout_e;

/**
 * @brief      Reload equality operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     equal or not
 */
bool operator==(const clcg_complex &a, const clcg_complex &b);

/**
 * @brief      Reload equality operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     unequal or not
 */
bool operator!=(const clcg_complex &a, const clcg_complex &b);

/**
 * @brief      Reload addition operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     sum
 */
clcg_complex operator+(const clcg_complex &a, const clcg_complex &b);

/**
 * @brief      Reload subtraction operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     subtraction
 */
clcg_complex operator-(const clcg_complex &a, const clcg_complex &b);

/**
 * @brief      Reload multiplication operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     product
 */
clcg_complex operator*(const clcg_complex &a, const clcg_complex &b);

/**
 * @brief      Reload division operator.
 *
 * @param[in]  a     complex number a
 * @param[in]  b     complex number b
 *
 * @return     quotient
 */
clcg_complex operator/(const clcg_complex &a, const clcg_complex &b);

/**
 * @brief      calculate complex conjugate
 *
 * @param[in]  a     complex number a
 *
 * @return     complex conjugate
 */
clcg_complex conjugate(const clcg_complex &a);

/**
 * @brief      calculate the product of a real number multiplied by a complex number
 *
 * @param[in]  a     real number a
 * @param[in]  b     complex number b
 *
 * @return     complex number
 */
clcg_complex real_product(const lcg_float &a, const clcg_complex &b);

/**
 * @brief      calculate inner product of two complex vectors
 * 
 * the product of two complex vectors are defined as <a, b> = \sum{\bar{a_i}\cdot\b_i}
 *
 * @param[in]  a       complex vector a
 * @param[in]  b       complex vector b
 * @param[in]  x_size  size of the vector
 *
 * @return     product
 */
clcg_complex inner_product(const clcg_complex *a, const clcg_complex *b, int x_size);

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
void matrix_product(clcg_complex **A, const clcg_complex *x, clcg_complex *Ax, 
	int m_size, int n_size, matrix_layout_e layout, bool conjugate = false);

/**
 * @brief      Types of method that could be recognized by the clcg_solver() function.
 */
typedef enum
{
	/**
	 * Jacob's Bi-conjugate Gradient Method
	 */
	CLCG_BICG,
	/**
	 * Conjugate Gradient Squared Method.
	 */
	CLCG_CGS,
} clcg_solver_enum;

/**
 * @brief      Parameters of the conjugate gradient methods.
 */
typedef struct
{
	/**
	 * Maximal iteration times. The default value is 100. one adjust this parameter 
	 * by passing a lcg_para type to the lcg_solver() function.
	*/
	int max_iterations;

	/**
	 * Epsilon for convergence test.
	 * This parameter determines the accuracy with which the solution is to be found. 
	 * A minimization terminates when ||g||/||b|| <= epsilon or |Ax - B| <= epsilon for 
	 * the lcg_solver() function, where ||.|| denotes the Euclidean (L2) norm and | | 
	 * denotes the L1 norm. The default value of epsilon is 1e-6. For box-constrained methods,
	 * the convergence test is implemented using ||P(m-g) - m|| <= epsilon, in which P is the
	 * projector that transfers m into the constrained domain.
	*/
	lcg_float epsilon;

	/**
	 * Whether to use absolute mean differences (AMD) between |Ax - B| to evaluate the process. 
	 * The default value is false which means the gradient based evaluating method is used. 
	 * The AMD based method will be used if this variable is set to true. This parameter is only 
	 * applied to the non-constrained methods.
	 */
	int abs_diff;
} clcg_para;

/**
 * @brief  Callback interface for calculating the complex product of a N*N matrix 'A' multiplied 
 * by a complex vertical vector 'x'.
 * 
 * @param  instance    The user data sent for the clcg_solver() functions by the client.
 * @param  x           Multiplier of the Ax product.
 * @param  Ax          Product of A multiplied by x.
 * @param  x_size      Size of x and column/row numbers of A.
 * @param  conjugate   Using the conjugate of A for calculation.
 */
typedef void (*clcg_axfunc_ptr)(void *instance, const clcg_complex *x, clcg_complex *prod_Ax, 
	const int x_size, bool conjugate);

/**
 * @brief     Callback interface for monitoring the progress and terminate the iteration 
 * if necessary.
 * 
 * @param    instance    The user data sent for the clcg_solver() functions by the client.
 * @param    m           The current solutions.
 * @param    converge    The current value evaluating the iteration progress.
 * @param    n_size      The size of the variables
 * @param    k           The iteration count.
 * 
 * @retval   int         Zero to continue the optimization process. Returning a
 *                       non-zero value will terminate the optimization process.
 */
typedef int (*clcg_progress_ptr)(void* instance, const clcg_complex* m, 
	const lcg_float converge, const clcg_para* param, const int n_size, const int k);

/**
 * @brief      Locate memory for a clcg_complex pointer type.
 *
 * @param[in]  n     Size of the lcg_float array.
 *
 * @return     Pointer of the array's location.
 */
clcg_complex* clcg_malloc(const int n);

/**
 * @brief      Destroy memory used by the clcg_complex type array.
 *
 * @param      x     Pointer of the array.
 */
void clcg_free(clcg_complex* x);

/**
 * @brief      Return a clcg_para type instance with default values.
 * 
 * Users can use this function to get default parameters' value for the complex conjugate gradient methods.
 * 
 * @return     A clcg_para type instance.
 */
clcg_para clcg_default_parameters();

/**
 * @brief      Return a string explanation for the clcg_solver() function's return values.
 *
 * @param[in]  er_index  The error index returned by the clcg_solver() function.
 *
 * @return     A string explanation of the error.
 */
const char* clcg_error_str(int er_index);

/**
 * @brief      A combined complex conjugate gradient solver function.
 *
 * @param[in]  Afp         Callback function for calculating the product of 'Ax'.
 * @param[in]  Pfp         Callback function for monitoring the iteration progress.
 * @param      m           Initial solution vector.
 * @param      B           Objective vector of the linear system.
 * @param[in]  n_size      Size of the solution vector and objective vector.
 * @param      param       Parameter setup for the conjugate gradient methods.
 * @param      instance    The user data sent for the lcg_solver() function by the client. 
 * This variable is either 'this' for class member functions or 'NULL' for global functions.
 * @param      solver_id   Solver type used to solve the linear system. The default value is LCG_CGS.
 *
 * @return     Status of the function.
 */
int clcg_solver(clcg_axfunc_ptr Afp, clcg_progress_ptr Pfp, clcg_complex* m, 
	const clcg_complex* B, const int n_size, const clcg_para* param, void* instance, 
	clcg_solver_enum solver_id = CLCG_CGS);

#ifdef __cplusplus
}
#endif

#endif //_CLCG_H