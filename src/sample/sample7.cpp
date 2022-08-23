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

#include "iostream"
#include "fstream"
#include "../lib/solver_eigen.h"
#include "../lib/preconditioner_eigen.h"

typedef std::complex<double> complex_d;
typedef Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> spmat_cd;
typedef Eigen::Triplet<complex_d> triplt_cd;
typedef Eigen::VectorXcd vector_cd;

void read(std::string filePath, int *pN, int *pnz, complex_d **cooVal, 
	int **cooRowIdx, int **cooColIdx, complex_d **b)
{
	std::ifstream in(filePath, std::ios::binary);

	in.read((char*)pN, sizeof(int));
	in.read((char*)pnz, sizeof(int));

	*cooVal = new complex_d[*pnz]{};
	*cooRowIdx = new int[*pnz]{};
	*cooColIdx = new int[*pnz]{};
	*b = new complex_d[*pN]{};

	for (int i = 0; i < *pnz; ++i)
	{
		in.read((char*)&(*cooRowIdx)[i], sizeof(int));
		in.read((char*)&(*cooColIdx)[i], sizeof(int));
		in.read((char*)&(*cooVal)[i], sizeof(complex_d));
	}

	in.read((char*)(*b), sizeof(complex_d)*(*pN));
    return;
}

void readAnswer(std::string filePath, int *pN, complex_d **x)
{
	std::ifstream in(filePath, std::ios::binary);

	in.read((char*)pN, sizeof(int));

	*x = new complex_d[*pN]{};

	in.read((char*)(*x), sizeof(complex_d)*(*pN));
    return;
}

double max_diff(const vector_cd &a, const vector_cd &b)
{
	double max = -1;
	complex_d t;
	for (int i = 0; i < a.size(); i++)
	{
		t = a[i] - b[i];
		max = lcg_max(std::sqrt(std::norm(t)), max);
	}
	return max;
}

class TESTFUNC : public CLCG_EIGEN_Solver
{
public:
	TESTFUNC(int n);
	~TESTFUNC();

	void set_kernel(int *row_id, int *col_id, complex_d *val, int nz_size);
	void set_preconditioner();

	//定义共轭梯度中Ax的算法
	void AxProduct(const vector_cd &x, vector_cd &prod_Ax, lcg_matrix_e layout, clcg_complex_e conjugate)
	{
		if (conjugate == Conjugate) prod_Ax = kernel.conjugate() * x;
		else prod_Ax = kernel * x;
		return;
	}

	void MxProduct(const vector_cd &x, vector_cd &prod_Mx, lcg_matrix_e layout, clcg_complex_e conjugate)
	{
		// No preconditioning
		//prod_Mx = x;

		// Preconditioning using the diagonal kernel
		//prod_Mx = p.cwiseProduct(x);

		// Preconditioning using the ILUT/IC
		clcg_solve_lower_triangle(l_tri, x, p);
		clcg_solve_upper_triangle(u_tri, p, prod_Mx);
		return;
	}

private:
	// 普通二维数组做核矩阵
	spmat_cd kernel, l_tri, u_tri;
	vector_cd p;
	int n_size;
};

TESTFUNC::TESTFUNC(int n)
{
	n_size = n;
	kernel.resize(n_size, n_size);
	kernel.setZero();
	p.resize(n_size);
}

TESTFUNC::~TESTFUNC()
{
	kernel.resize(0, 0);
	l_tri.resize(0, 0);
	u_tri.resize(0, 0);
	p.resize(0);
}

void TESTFUNC::set_kernel(int *row_id, int *col_id, complex_d *val, int nz_size)
{
	std::vector<triplt_cd> val_triplt;
	for (size_t i = 0; i < nz_size; i++)
	{
		val_triplt.push_back(triplt_cd(row_id[i], col_id[i], val[i]));
	}

	kernel.setFromTriplets(val_triplt.begin(), val_triplt.end());
	return;
}

void TESTFUNC::set_preconditioner()
{
	// 1 Preconditioning using the incomplete LU decomposition
	/*
	for (size_t i = 0; i < n_size; i++)
	{
		p[i] = 1.0/kernel.coeff(i, i);
	}
	*/

	// 2. Preconditioning using the incomplete LU decomposition
	//incomplete_LU(kernel, l_tri, u_tri);

	// 3. Preconditioning using the incomplete Cholesky decomposition
	clcg_incomplete_Cholesky(kernel, l_tri);
	u_tri = l_tri.transpose();

	// 4. Preconditioning using compressed incomplete decompositions
	/*
	vector_cd one = Eigen::VectorXcd::Ones(n_size);
	vector_cd x = Eigen::VectorXcd::Zero(n_size);

	solve_lower_triangle(l_tri, one, x);
	solve_upper_triangle(u_tri, x, p);
	*/
	return;
}

int main(int argc, char const *argv[]) try
{
	std::string inputPath = "data/case_1K_cA";
	std::string answerPath = "data/case_1K_cB";

	int N;
	int nz;
	complex_d *A;
	int *rowIdxA;
	int *colIdxA;
	complex_d *b;
	read(inputPath, &N, &nz, &A, &rowIdxA, &colIdxA, &b);

	complex_d *ans_x;
	readAnswer(answerPath, &N, &ans_x);

	std::clog << "N = " << N << std::endl;
	std::clog << "nz = " << nz << std::endl;

	TESTFUNC test(N);
	test.set_kernel(rowIdxA, colIdxA, A, nz);
	test.set_preconditioner();

	vector_cd B, ANS;
	B.resize(N);
	ANS.resize(N);
	for (size_t i = 0; i < N; i++)
	{
		B[i] = b[i];
		ANS[i] = ans_x[i];
	}

	/********************准备工作完成************************/

	clcg_para self_para = clcg_default_parameters();
	self_para.epsilon = 1e-6;
	self_para.abs_diff = 1;
	test.set_clcg_parameter(self_para);
	test.set_report_interval(10);

	Eigen::VectorXcd m = Eigen::VectorXcd::Constant(N, std::complex<double>(0.0, 0.0));

	test.MinimizePreconditioned(m, B, CLCG_PCG);
	std::clog << "maximal difference: " << max_diff(ANS, m) << std::endl << std::endl;

	m.setZero();
	test.MinimizePreconditioned(m, B, CLCG_PBICG);
	std::clog << "maximal difference: " << max_diff(ANS, m) << std::endl << std::endl;

	ANS.resize(0);
	B.resize(0);
	m.resize(0);

	return 0;
}
catch (std::exception &e)
{
	std::cerr << e.what() << std::endl;
}