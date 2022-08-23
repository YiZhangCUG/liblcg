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

#ifndef _PRECONDITIONER_EIGEN_H
#define _PRECONDITIONER_EIGEN_H

#include "complex"
#include "Eigen/Dense"
#include "Eigen/SparseCore"


/**
 * @brief     Perform the Cholesky decomposition and return the lower triangular matrix.
 * 
 * @note      This could serve as a direct solver.
 * 
 * @param A   The input matrix. Must be full rank and symmetric (aka. A = A^T)
 * @param L   The output low triangular matrix
 */
void lcg_Cholesky(const Eigen::MatrixXd &A, Eigen::MatrixXd &L);

/**
 * @brief      Perform the Cholesky decomposition and return the lower triangular matrix
 * 
 * @note       This could serve as a direct solver.
 *
 * @param[in]  A     The input matrix. Must be full rank and symmetric (aka. A = A^T)
 * @param      L     The output low triangular matrix
 */
void clcg_Cholesky(const Eigen::MatrixXcd &A, Eigen::MatrixXcd &L);

/**
 * @brief      Calculate the invert of a lower triangle matrix (Full rank only).
 *
 * @param      L     The operating lower triangle matrix
 * @param      Linv  The inverted lower triangle matrix
 */
void lcg_invert_lower_triangle(const Eigen::MatrixXd &L, Eigen::MatrixXd &Linv);

/**
 * @brief      Calculate the invert of a upper triangle matrix (Full rank only).
 *
 * @param      U     The operating upper triangle matrix
 * @param      Uinv  The inverted upper triangle matrix
 */
void lcg_invert_upper_triangle(const Eigen::MatrixXd &U, Eigen::MatrixXd &Uinv);

/**
 * @brief      Calculate the invert of a lower triangle matrix (Full rank only).
 *
 * @param      L     The operating lower triangle matrix
 * @param      Linv  The inverted lower triangle matrix
 */
void clcg_invert_lower_triangle(const Eigen::MatrixXcd &L, Eigen::MatrixXcd &Linv);

/**
 * @brief      Calculate the invert of a upper triangle matrix (Full rank only).
 *
 * @param      U     The operating upper triangle matrix
 * @param      Uinv  The inverted upper triangle matrix
 */
void clcg_invert_upper_triangle(const Eigen::MatrixXcd &U, Eigen::MatrixXcd &Uinv);

/**
 * @brief      Calculate the incomplete Cholesky decomposition and return the lower triangular matrix
 *
 * @param[in]  A     The input sparse matrix. Must be full rank and symmetric (aka. A = A^T)
 * @param      L     The output lower triangular matrix
 * @param      fill  The fill-in number of the output sparse matrix. No fill-in reduction will be processed if this variable is set to zero.
 */
void lcg_incomplete_Cholesky(const Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::SparseMatrix<double, Eigen::RowMajor> &L, size_t fill = 0);

/**
 * @brief      Calculate the incomplete Cholesky decomposition and return the lower triangular matrix
 *
 * @param[in]  A     The input sparse matrix. Must be full rank and symmetric (aka. A = A^T)
 * @param      L     The output lower triangular matrix
 * @param      fill  The fill-in number of the output sparse matrix. No fill-in reduction will be processed if this variable is set to zero.
 */
void clcg_incomplete_Cholesky(const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &A, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &L, size_t fill = 0);

/**
 * @brief        Calculate the incomplete LU factorizations
 * 
 * @param A      The input sparse matrix. Must be full rank.
 * @param L      The output lower triangular matrix.
 * @param U      The output upper triangular matrix.
 * @param fill   The fill-in number of the output sparse matrix. No fill-in reduction will be processed if this variable is set to zero.
 */
void lcg_incomplete_LU(const Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::SparseMatrix<double, Eigen::RowMajor> &L, Eigen::SparseMatrix<double, Eigen::RowMajor> &U, size_t fill = 0);

/**
 * @brief        Calculate the incomplete LU factorizations
 * 
 * @param A      The input sparse matrix. Must be full rank.
 * @param L      The output lower triangular matrix.
 * @param U      The output upper triangular matrix.
 * @param fill   The fill-in number of the output sparse matrix. No fill-in reduction will be processed if this variable is set to zero.
 */
void clcg_incomplete_LU(const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &A, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &L, 
    Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &U, size_t fill = 0);

/**
 * @brief    Solve the linear system Lx = B, in which L is a lower triangle matrix.
 * 
 * @param L  The input lower triangle matrix
 * @param B  The object vector
 * @param X  The solution vector
 */
void lcg_solve_lower_triangle(const Eigen::SparseMatrix<double, Eigen::RowMajor> &L, const Eigen::VectorXd &B, Eigen::VectorXd &X);

/**
 * @brief    Solve the linear system Ux = B, in which U is a upper triangle matrix.
 * 
 * @param U  The input upper triangle matrix
 * @param B  The object vector
 * @param X  The solution vector
 */
void lcg_solve_upper_triangle(const Eigen::SparseMatrix<double, Eigen::RowMajor> &U, const Eigen::VectorXd &B, Eigen::VectorXd &X);

/**
 * @brief    Solve the linear system Lx = B, in which L is a lower triangle matrix.
 * 
 * @param L  The input lower triangle matrix
 * @param B  The object vector
 * @param X  The solution vector
 */
void clcg_solve_lower_triangle(const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &L, const Eigen::VectorXcd &B, Eigen::VectorXcd &X);

/**
 * @brief    Solve the linear system Ux = B, in which U is a upper triangle matrix.
 * 
 * @param U  The input upper triangle matrix
 * @param B  The object vector
 * @param X  The solution vector
 */
void clcg_solve_upper_triangle(const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &U, const Eigen::VectorXcd &B, Eigen::VectorXcd &X);


#endif // _PRECONDITIONER_EIGEN_H