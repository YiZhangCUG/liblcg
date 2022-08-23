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

#include "preconditioner_eigen.h"

#include "exception"
#include "stdexcept"
#include "vector"
#include "iostream"

#ifdef LibLCG_OPENMP
#include "omp.h"
#endif


typedef Eigen::Triplet<int> triplet_bl;
typedef Eigen::Triplet<double> triplet_d;
typedef Eigen::Triplet<std::complex<double> > triplet_cd;

void lcg_Cholesky(const Eigen::MatrixXd &A, Eigen::MatrixXd &L)
{
	size_t num = A.rows();
/*
	if (A.rows() != A.cols()) throw std::runtime_error("The input matrix is not square. From Cholesky(...).");
	
	for (size_t i = 0; i < num; i++)
	{
		if (A.coeff(i, i) == 0.0)
			throw std::runtime_error("The input matrix is not full-rank. From Cholesky(...).");
	}
*/
	L.resize(num, num);
	L.setZero();
	// Copy the lower triangle part of A to L
	for (size_t j = 0; j < num; j++)
	{
		for (size_t i = j; i < num; i++)
		{
			L.coeffRef(i, j) = A.coeff(i, j);
		}
	}

	// Calculate the first column
	L.coeffRef(0, 0) = std::sqrt(A.coeff(0, 0));
	for (size_t i = 1; i < num; i++)
	{
		L.coeffRef(i, 0) = A.coeff(i, 0)/L.coeff(0, 0);
	}

	double Lnorm;
	for (size_t j = 0; j < num-1; j++)
	{
		Lnorm = 0.0;
		for (size_t i = 0; i < j+1; i++)
		{
			Lnorm += L.coeff(j+1, i) * L.coeff(j+1, i);
		}

		L.coeffRef(j+1, j+1) = std::sqrt(A.coeff(j+1, j+1) - Lnorm);

		for (size_t i = j+1; i < num-1; i++)
		{
			for (size_t k = i+1; k < num; k++)
			{
				L.coeffRef(k, i) -= L.coeff(k, j) * L.coeff(i, j);
			}
		}

		for (size_t i = j+2; i < num; i++)
		{
			L.coeffRef(i, j+1) /= L.coeff(j+1, j+1);
		}
	}
	return;
}

void clcg_Cholesky(const Eigen::MatrixXcd &A, Eigen::MatrixXcd &L)
{
	size_t num = A.rows();
/*
	if (A.rows() != A.cols()) throw std::runtime_error("The input matrix is not square. From Cholesky(...).");
	
	for (size_t i = 0; i < num; i++)
	{
		if (A.coeff(i, i) == std::complex<double>(0.0, 0.0))
			throw std::runtime_error("The input matrix is not full-rank. From Cholesky(...).");
	}
*/
	L.resize(num, num);
	L.setZero();
	// Copy the lower triangle part of A to L
	for (size_t j = 0; j < num; j++)
	{
		for (size_t i = j; i < num; i++)
		{
			L.coeffRef(i, j) = A.coeff(i, j);
		}
	}

	// Calculate the first column
	L.coeffRef(0, 0) = std::sqrt(A.coeff(0, 0));
	for (size_t i = 1; i < num; i++)
	{
		L.coeffRef(i, 0) = A.coeff(i, 0)/L.coeff(0, 0);
	}

	std::complex<double> Lnorm;
	for (size_t j = 0; j < num-1; j++)
	{
		Lnorm = std::complex<double>(0.0, 0.0);
		for (size_t i = 0; i < j+1; i++)
		{
			Lnorm += L.coeff(j+1, i) * L.coeff(j+1, i);
		}

		L.coeffRef(j+1, j+1) = std::sqrt(A.coeff(j+1, j+1) - Lnorm);

		for (size_t i = j+1; i < num-1; i++)
		{
			for (size_t k = i+1; k < num; k++)
			{
				L.coeffRef(k, i) -= L.coeff(k, j) * L.coeff(i, j);
			}
		}

		for (size_t i = j+2; i < num; i++)
		{
			L.coeffRef(i, j+1) /= L.coeff(j+1, j+1);
		}
	}
	return;
}

void lcg_invert_lower_triangle(const Eigen::MatrixXd &L, Eigen::MatrixXd &Linv)
{
	if (L.rows() != L.cols()) throw std::runtime_error("The input matrix is not square. From invert_lower_triangle(...).");

	size_t num = L.rows();
	for (size_t i = 0; i < num; i++)
	{
		if (L.coeff(i, i) == 0.0)
			throw std::runtime_error("The input matrix is not full-rank. From invert_lower_triangle(...).");
	}

	// modify the diagonal elements
	Linv.resize(num, num);
	Linv.setZero();
	for (size_t i = 0; i < num; i++)
	{
		Linv.coeffRef(i, i) = 1.0/L.coeff(i, i);
	}

	// calculate other elements column by column
	double sum;
	for (size_t j = 0; j < num-1; j++)
	{
		for (size_t i = j+1; i < num; i++)
		{
			sum = L.coeff(i, j)/L.coeff(j, j);
			for (size_t k = j+1; k < i; k++)
			{
				sum += L.coeff(i, k) * Linv.coeff(k, j);
			}
			Linv.coeffRef(i, j) = -1.0*sum/L.coeff(i, i);
		}
	}
	return;
}

void lcg_invert_upper_triangle(const Eigen::MatrixXd &U, Eigen::MatrixXd &Uinv)
{
	if (U.rows() != U.cols()) throw std::runtime_error("The input matrix is not square. From invert_upper_triangle(...).");

	size_t num = U.rows();
	for (size_t i = 0; i < num; i++)
	{
		if (U.coeff(i, i) == 0.0)
			throw std::runtime_error("The input matrix is not full-rank. From invert_upper_triangle(...).");
	}

	// modify the diagonal elements
	Uinv.resize(num, num);
	Uinv.setZero();
	for (size_t i = 0; i < num; i++)
	{
		Uinv.coeffRef(i, i) = 1.0/U.coeff(i, i);
	}

	// calculate other elements column by column
	double sum;
	for (size_t j = num-1; j > 0; j--)
	{
		for (int i = j-1; i >= 0; i--)
		{
			sum = U.coeff(i, j)/U.coeff(j, j);
			for (size_t k = j-1; k > i; k--)
			{
				sum += U.coeff(i, k) * Uinv.coeff(k, j);
			}
			Uinv.coeffRef(i, j) = -1.0*sum/U.coeff(i, i);
		}
	}
	return;
}

void clcg_invert_lower_triangle(const Eigen::MatrixXcd &L, Eigen::MatrixXcd &Linv)
{
	if (L.rows() != L.cols()) throw std::runtime_error("The input matrix is not square. From invert_lower_triangle(...).");

	size_t num = L.rows();
	for (size_t i = 0; i < num; i++)
	{
		if (L.coeff(i, i) == std::complex<double>(0.0, 0.0))
			throw std::runtime_error("The input matrix is not full-rank. From invert_lower_triangle(...).");
	}

	// modify the diagonal elements
	Linv.resize(num, num);
	Linv.setZero();
	for (size_t i = 0; i < num; i++)
	{
		Linv.coeffRef(i, i) = 1.0/L.coeff(i, i);
	}

	// calculate other elements column by column
	std::complex<double> sum;
	for (size_t j = 0; j < num-1; j++)
	{
		for (size_t i = j+1; i < num; i++)
		{
			sum = L.coeff(i, j)/L.coeff(j, j);
			for (size_t k = j+1; k < i; k++)
			{
				sum += L.coeff(i, k) * Linv.coeff(k, j);
			}
			Linv.coeffRef(i, j) = -1.0*sum/L.coeff(i, i);
		}
	}
	return;
}

void clcg_invert_upper_triangle(const Eigen::MatrixXcd &U, Eigen::MatrixXcd &Uinv)
{
	if (U.rows() != U.cols()) throw std::runtime_error("The input matrix is not square. From invert_upper_triangle(...).");

	size_t num = U.rows();
	for (size_t i = 0; i < num; i++)
	{
		if (U.coeff(i, i) == std::complex<double>(0.0, 0.0))
			throw std::runtime_error("The input matrix is not full-rank. From invert_upper_triangle(...).");
	}

	// modify the diagonal elements
	Uinv.resize(num, num);
	Uinv.setZero();
	for (size_t i = 0; i < num; i++)
	{
		Uinv.coeffRef(i, i) = 1.0/U.coeff(i, i);
	}

	// calculate other elements column by column
	std::complex<double> sum;
	for (size_t j = num-1; j > 0; j--)
	{
		for (int i = j-1; i >= 0; i--)
		{
			sum = U.coeff(i, j)/U.coeff(j, j);
			for (size_t k = j-1; k > i; k--)
			{
				sum += U.coeff(i, k) * Uinv.coeff(k, j);
			}
			Uinv.coeffRef(i, j) = -1.0*sum/U.coeff(i, i);
		}
	}
	return;
}

void lcg_incomplete_Cholesky(const Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::SparseMatrix<double, Eigen::RowMajor> &L, size_t fill)
{
	size_t i, j, k;
	size_t num = A.rows();
	/*
	if (A.rows() != A.cols()) throw std::runtime_error("The input matrix is not square. From incomplete_Cholesky(...).");

	for (i = 0; i < num; i++)
	{
		if (A.coeff(i, i) == 0.0)
			throw std::runtime_error("The input matrix is not full-rank. From incomplete_Cholesky(...).");
	}
	*/

	Eigen::SparseMatrix<int, Eigen::ColMajor> L_graph;
	std::vector<triplet_d> L_entries;
	std::vector<triplet_bl> G_entries;
	Eigen::VectorXd row(num);
	row.setZero();

	L.resize(num, num);
	L_graph.resize(num, num);
	// Copy the lower triangle part of A to L
	if (fill == 0)
	{
		for (i = 0; i < num; i++)
		{
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() <= it.row())
				{
					L_entries.push_back(triplet_d(i, it.col(), it.value()));
					if (it.col() != it.row()) G_entries.push_back(triplet_bl(i, it.col(), 1));
				}
			}
		}
	}
	else
	{
		int nz_count, mini_id;
		double mini_val;
		for (i = 0; i < num; i++)
		{
			nz_count = 0;
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() <= it.row())
				{
					row[it.col()] = it.value();
					nz_count++;
				}
			}

			while (nz_count > fill)
			{
				mini_val = 1e+30;
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
				{
					if (it.col() < it.row() && row[it.col()] != 0.0 && std::norm(row[it.col()]) < mini_val)
					{
						mini_val = std::norm(row[it.col()]);
						mini_id = it.col();
					}
				}

				row[mini_id] = 0.0;
				nz_count--;
			}

			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() <= it.row() && row[it.col()] != 0.0)
				{
					L_entries.push_back(triplet_d(i, it.col(), row[it.col()]));
					if (it.col() != it.row()) G_entries.push_back(triplet_bl(i, it.col(), 1));
					row[it.col()] = 0.0;
				}
			}
		}
	}

	L.setFromTriplets(L_entries.begin(), L_entries.end());
	L_graph.setFromTriplets(G_entries.begin(), G_entries.end());
	L_entries.clear();
	G_entries.clear();

	// Calculate the first column
	L.coeffRef(0, 0) = std::sqrt(L.coeff(0, 0));
	double factor = 1.0/L.coeff(0, 0);
	for (Eigen::SparseMatrix<int>::InnerIterator ig(L_graph, 0); ig; ++ig)
	{
		L.coeffRef(ig.row(), 0) = factor*L.coeff(ig.row(), 0);
	}

	double Ljj, sum;
	for (j = 1; j < num; j++)
	{
		sum = 0.0;
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L, j); it; ++it)
		{
			if (it.col() < j)
			{
				sum += it.value() * it.value();
				row[it.col()] = it.value();
			}
			else if (it.col() == j) // must exit
			{
				it.valueRef() = std::sqrt(it.value() - sum);
				Ljj = it.value();
				break;
			}
		}

		for (Eigen::SparseMatrix<int>::InnerIterator ig(L_graph, j); ig; ++ig)
		{
			sum = 0.0;
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L, ig.row()); it; ++it)
			{
				if (it.col() < j)
				{
					sum += row[it.col()] * it.value();
				}
				else if (it.col() == j)
				{
					it.valueRef() = (it.value() - sum)/Ljj;
					break;
				}
			}
		}

		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L, j); it; ++it)
		{
			if (it.col() < j) row[it.col()] = 0.0;
		}
	}

	L_graph.resize(0, 0);
	return;
}

void clcg_incomplete_Cholesky(const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &A, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &L, size_t fill)
{
	size_t i, j, k;
	size_t num = A.rows();
	std::complex<double> zero = std::complex<double>(0.0, 0.0);
	/*
	if (A.rows() != A.cols()) throw std::runtime_error("The input matrix is not square. From incomplete_Cholesky(...).");

	for (i = 0; i < num; i++)
	{
		if (A.coeff(i, i) == zero)
			throw std::runtime_error("The input matrix is not full-rank. From incomplete_Cholesky(...).");
	}
	*/

	Eigen::SparseMatrix<int, Eigen::ColMajor> L_graph;
	std::vector<triplet_cd> L_entries;
	std::vector<triplet_bl> G_entries;
	Eigen::VectorXcd row(num);
	row.setZero();

	L.resize(num, num);
	L_graph.resize(num, num);
	// Copy the lower triangle part of A to L
	if (fill == 0)
	{
		for (i = 0; i < num; i++)
		{
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() <= it.row())
				{
					L_entries.push_back(triplet_cd(i, it.col(), it.value()));
					if (it.col() != it.row()) G_entries.push_back(triplet_bl(i, it.col(), 1));
				}
			}
		}
	}
	else
	{
		int nz_count, mini_id;
		double mini_val;
		for (i = 0; i < num; i++)
		{
			nz_count = 0;
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() <= it.row())
				{
					row[it.col()] = it.value();
					nz_count++;
				}
			}

			while (nz_count > fill)
			{
				mini_val = 1e+30;
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
				{
					if (it.col() < it.row() && row[it.col()] != zero && std::norm(row[it.col()]) < mini_val)
					{
						mini_val = std::norm(row[it.col()]);
						mini_id = it.col();
					}
				}

				row[mini_id] = zero;
				nz_count--;
			}

			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() <= it.row() && row[it.col()] != zero)
				{
					L_entries.push_back(triplet_cd(i, it.col(), row[it.col()]));
					if (it.col() != it.row()) G_entries.push_back(triplet_bl(i, it.col(), 1));
					row[it.col()] = zero;
				}
			}
		}
	}

	L.setFromTriplets(L_entries.begin(), L_entries.end());
	L_graph.setFromTriplets(G_entries.begin(), G_entries.end());
	L_entries.clear();
	G_entries.clear();

	// Calculate the first column
	L.coeffRef(0, 0) = std::sqrt(L.coeff(0, 0));
	std::complex<double> factor = 1.0/L.coeff(0, 0);
	for (Eigen::SparseMatrix<int>::InnerIterator ig(L_graph, 0); ig; ++ig)
	{
		L.coeffRef(ig.row(), 0) = factor*L.coeff(ig.row(), 0);
	}

	std::complex<double> Ljj, sum;
	for (j = 1; j < num; j++)
	{
		sum = zero;
		for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(L, j); it; ++it)
		{
			if (it.col() < j)
			{
				sum += it.value() * it.value();
				row[it.col()] = it.value();
			}
			else if (it.col() == j) // must exit
			{
				it.valueRef() = std::sqrt(it.value() - sum);
				Ljj = it.value();
				break;
			}
		}

		for (Eigen::SparseMatrix<int>::InnerIterator ig(L_graph, j); ig; ++ig)
		{
			sum = zero;
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(L, ig.row()); it; ++it)
			{
				if (it.col() < j)
				{
					sum += row[it.col()] * it.value();
				}
				else if (it.col() == j)
				{
					it.valueRef() = (it.value() - sum)/Ljj;
					break;
				}
			}
		}

		for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(L, j); it; ++it)
		{
			if (it.col() < j) row[it.col()] = zero;
		}
	}

	L_graph.resize(0, 0);
	return;
}

void lcg_incomplete_LU(const Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::SparseMatrix<double, Eigen::RowMajor> &L, Eigen::SparseMatrix<double, Eigen::RowMajor> &U, size_t fill)
{
	size_t i, j, k;
	size_t num = A.rows();
	/*
	if (A.rows() != A.cols()) throw std::runtime_error("The input matrix is not square. From incomplete_LU(...).");

	for (i = 0; i < num; i++)
	{
		if (A.coeff(i, i) == 0.0)
			throw std::runtime_error("The input matrix is not full-rank. From incomplete_LU(...).");
	}
	*/

	Eigen::SparseMatrix<double, Eigen::RowMajor> T;
	std::vector<triplet_d> T_entries;

	Eigen::VectorXd row(num);
	row.setZero();

	// Copy the input matrix to a temporary matrix
	if (fill == 0) T = A;
	else
	{
		int nz_count, mini_id;
		double mini_val;
		for (i = 0; i < num; i++)
		{
			nz_count = 0;
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() <= it.row())
				{
					row[it.col()] = it.value();
					nz_count++;
				}
			}

			while (nz_count > fill+1)
			{
				mini_val = 1e+30;
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
				{
					if (it.col() < it.row() && row[it.col()] != 0.0 && std::norm(row[it.col()]) < mini_val)
					{
						mini_val = std::norm(row[it.col()]);
						mini_id = it.col();
					}
				}

				row[mini_id] = 0.0;
				nz_count--;
			}

			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() <= it.row() && row[it.col()] != 0.0)
				{
					T_entries.push_back(triplet_d(i, it.col(), row[it.col()]));
					row[it.col()] = 0.0;
				}
			}

			nz_count = 0;
			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() > it.row())
				{
					row[it.col()] = it.value();
					nz_count++;
				}
			}

			while (nz_count > fill)
			{
				mini_val = 1e+30;
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
				{
					if (it.col() > it.row() && row[it.col()] != 0.0 && std::norm(row[it.col()]) < mini_val)
					{
						mini_val = std::norm(row[it.col()]);
						mini_id = it.col();
					}
				}

				row[mini_id] = 0.0;
				nz_count--;
			}

			for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() > it.row() && row[it.col()] != 0.0)
				{
					T_entries.push_back(triplet_d(i, it.col(), row[it.col()]));
					row[it.col()] = 0.0;
				}
			}
		}

		T.resize(num, num);
		T.setFromTriplets(T_entries.begin(), T_entries.end());
		T_entries.clear();
	}

	for (i = 1; i < num; i++)
	{
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(T, i); it; ++it)
		{
			if (it.col() < it.row()) // Operate on the lower half triangle
			{
				// locate the dependent row and copy entries
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator ik(T, it.col()); ik; ++ik)
				{
					if (ik.col() >= ik.row())
					{
						row[ik.col()] = ik.value();
					}
				}

				it.valueRef() = it.value()/row[it.col()];

				// Operate on the remaining elements in the row
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator ir(T, i); ir; ++ir)
				{
					if (ir.col() > it.col())
					{
						ir.valueRef() = ir.value() - it.value()*row[ir.col()];
					}
				}

				// Reset temporary row
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator ik(T, it.col()); ik; ++ik)
				{
					if (ik.col() >= ik.row())
					{
						row[ik.col()] = 0.0;
					}
				}
			}
		}
	}

	L.resize(num, num);
	U.resize(num, num);
	std::vector<triplet_d> L_entries;
	std::vector<triplet_d> U_entries;

	for (i = 0; i < num; i++)
	{
		L_entries.push_back(triplet_d(i, i, 1.0));
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(T, i); it; ++it)
		{
			if (it.col() < it.row())
			{
				L_entries.push_back(triplet_d(i, it.col(), it.value()));
			}
			else
			{
				U_entries.push_back(triplet_d(i, it.col(), it.value()));
			}
		}
	}

	L.setFromTriplets(L_entries.begin(), L_entries.end());
	U.setFromTriplets(U_entries.begin(), U_entries.end());
	L_entries.clear();
	U_entries.clear();

	T.resize(0, 0);
	return;
}

void clcg_incomplete_LU(const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &A, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &L, 
    Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &U, size_t fill)
{
	size_t i, j, k;
	size_t num = A.rows();
	std::complex<double> zero = std::complex<double>(0.0, 0.0);
	std::complex<double> one  = std::complex<double>(1.0, 0.0);
	/*
	if (A.rows() != A.cols()) throw std::runtime_error("The input matrix is not square. From incomplete_LU(...).");

	for (i = 0; i < num; i++)
	{
		if (A.coeff(i, i) == zero)
			throw std::runtime_error("The input matrix is not full-rank. From incomplete_LU(...).");
	}
	*/

	Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> T;
	std::vector<triplet_cd> T_entries;

	Eigen::VectorXcd row(num);
	row.setZero();

	// Copy the input matrix to a temporary matrix
	if (fill == 0) T = A;
	else
	{
		int nz_count, mini_id;
		double mini_val;
		for (i = 0; i < num; i++)
		{
			nz_count = 0;
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() <= it.row())
				{
					row[it.col()] = it.value();
					nz_count++;
				}
			}

			while (nz_count > fill+1)
			{
				mini_val = 1e+30;
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
				{
					if (it.col() < it.row() && row[it.col()] != zero && std::norm(row[it.col()]) < mini_val)
					{
						mini_val = std::norm(row[it.col()]);
						mini_id = it.col();
					}
				}

				row[mini_id] = zero;
				nz_count--;
			}

			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() <= it.row() && row[it.col()] != zero)
				{
					T_entries.push_back(triplet_cd(i, it.col(), row[it.col()]));
					row[it.col()] = zero;
				}
			}

			nz_count = 0;
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() > it.row())
				{
					row[it.col()] = it.value();
					nz_count++;
				}
			}

			while (nz_count > fill)
			{
				mini_val = 1e+30;
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
				{
					if (it.col() > it.row() && row[it.col()] != zero && std::norm(row[it.col()]) < mini_val)
					{
						mini_val = std::norm(row[it.col()]);
						mini_id = it.col();
					}
				}

				row[mini_id] = zero;
				nz_count--;
			}

			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
			{
				if (it.col() > it.row() && row[it.col()] != zero)
				{
					T_entries.push_back(triplet_cd(i, it.col(), row[it.col()]));
					row[it.col()] = zero;
				}
			}
		}

		T.resize(num, num);
		T.setFromTriplets(T_entries.begin(), T_entries.end());
		T_entries.clear();
	}

	for (i = 1; i < num; i++)
	{
		for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(T, i); it; ++it)
		{
			if (it.col() < it.row()) // Operate on the lower half triangle
			{
				// locate the dependent row and copy entries
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator ik(T, it.col()); ik; ++ik)
				{
					if (ik.col() >= ik.row())
					{
						row[ik.col()] = ik.value();
					}
				}

				it.valueRef() = it.value()/row[it.col()];

				// Operate on the remaining elements in the row
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator ir(T, i); ir; ++ir)
				{
					if (ir.col() > it.col())
					{
						ir.valueRef() = ir.value() - it.value()*row[ir.col()];
					}
				}

				// Reset temporary row
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator ik(T, it.col()); ik; ++ik)
				{
					if (ik.col() >= ik.row())
					{
						row[ik.col()] = zero;
					}
				}
			}
		}
	}

	L.resize(num, num);
	U.resize(num, num);
	std::vector<triplet_cd> L_entries;
	std::vector<triplet_cd> U_entries;

	for (i = 0; i < num; i++)
	{
		L_entries.push_back(triplet_cd(i, i, one));
		for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(T, i); it; ++it)
		{
			if (it.col() < it.row())
			{
				L_entries.push_back(triplet_cd(i, it.col(), it.value()));
			}
			else
			{
				U_entries.push_back(triplet_cd(i, it.col(), it.value()));
			}
		}
	}

	L.setFromTriplets(L_entries.begin(), L_entries.end());
	U.setFromTriplets(U_entries.begin(), U_entries.end());
	L_entries.clear();
	U_entries.clear();

	T.resize(0, 0);
	return;
}

void lcg_solve_lower_triangle(const Eigen::SparseMatrix<double, Eigen::RowMajor> &L, const Eigen::VectorXd &B, Eigen::VectorXd &X)
{
	size_t num = L.rows();
/*
	if (L.rows() != L.cols()) throw std::runtime_error("The input matrix is not square. From solve_lower_triangle(...).");

	for (size_t i = 0; i < num; i++)
	{
		if (L.coeff(i, i) == 0.0)
			throw std::runtime_error("The input matrix is not full-rank. From solve_lower_triangle(...).");
	}

	if (num != B.size() || num != X.size())
	{
		throw std::runtime_error("The input vectors are of the wrong size. From solve_lower_triangle(...).");
	}
*/
	double sum, Lii;
	for (size_t i = 0; i < num; i++)
	{
		sum = 0.0;
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L, i); it; ++it)
		{
			if (it.col() < it.row()) sum += X[it.col()] * it.value();
			else if (it.col() == it.row()) Lii = it.value();
		}
		X[i] = (B[i] - sum)/Lii;
	}
	return;
}

void lcg_solve_upper_triangle(const Eigen::SparseMatrix<double, Eigen::RowMajor> &U, const Eigen::VectorXd &B, Eigen::VectorXd &X)
{
	size_t num = U.rows();
/*
	if (U.rows() != U.cols()) throw std::runtime_error("The input matrix is not square. From solve_upper_triangle(...).");

	for (size_t i = 0; i < num; i++)
	{
		if (U.coeff(i, i) == 0.0)
			throw std::runtime_error("The input matrix is not full-rank. From solve_upper_triangle(...).");
	}

	if (num != B.size() || num != X.size())
	{
		throw std::runtime_error("The input vectors are of the wrong size. From solve_upper_triangle(...).");
	}
*/
	double sum, Uii;
	for (int i = num-1; i >= 0; i--)
	{
		sum = 0.0;
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(U, i); it; ++it)
		{
			if (it.col() > it.row()) sum += X[it.col()] * it.value();
			else if (it.col() == it.row()) Uii = it.value();
		}
		X[i] = (B[i] - sum)/Uii;
	}
	return;
}

void clcg_solve_lower_triangle(const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &L, const Eigen::VectorXcd &B, Eigen::VectorXcd &X)
{
	size_t num = L.rows();
/*
	if (L.rows() != L.cols()) throw std::runtime_error("The input matrix is not square. From solve_lower_triangle(...).");

	for (size_t i = 0; i < num; i++)
	{
		if (L.coeff(i, i) == std::complex<double>(0.0, 0.0))
			throw std::runtime_error("The input matrix is not full-rank. From solve_lower_triangle(...).");
	}

	if (num != B.size() || num != X.size())
	{
		throw std::runtime_error("The input vectors are of the wrong size. From solve_lower_triangle(...).");
	}
*/
	std::complex<double> sum, Lii;
	for (size_t i = 0; i < num; i++)
	{
		sum = std::complex<double>(0.0, 0.0);
		for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(L, i); it; ++it)
		{
			if (it.col() < it.row()) sum += X[it.col()] * it.value();
			else if (it.col() == it.row()) Lii = it.value();
		}
		X[i] = (B[i] - sum)/Lii;
	}
	return;
}

void clcg_solve_upper_triangle(const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &U, const Eigen::VectorXcd &B, Eigen::VectorXcd &X)
{
	size_t num = U.rows();
/*
	if (U.rows() != U.cols()) throw std::runtime_error("The input matrix is not square. From solve_upper_triangle(...).");

	for (size_t i = 0; i < num; i++)
	{
		if (U.coeff(i, i) == std::complex<double>(0.0, 0.0))
			throw std::runtime_error("The input matrix is not full-rank. From solve_upper_triangle(...).");
	}

	if (num != B.size() || num != X.size())
	{
		throw std::runtime_error("The input vectors are of the wrong size. From solve_upper_triangle(...).");
	}
*/
	std::complex<double> sum, Uii;
	for (int i = num-1; i >= 0; i--)
	{
		sum = std::complex<double>(0.0, 0.0);
		for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(U, i); it; ++it)
		{
			if (it.col() > it.row()) sum += X[it.col()] * it.value();
			else if (it.col() == it.row()) Uii = it.value();
		}
		X[i] = (B[i] - sum)/Uii;
	}
	return;
}