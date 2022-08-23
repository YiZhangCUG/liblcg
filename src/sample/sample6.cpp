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
#include "complex"
#include "../lib/lcg_complex.h"
#include "../lib/solver_eigen.h"
#include "Eigen/Sparse"

typedef Eigen::SparseMatrix<lcg_complex, Eigen::RowMajor> spmat_cd; // 注意Eigen默认的稀疏矩阵排序为列优先
typedef Eigen::Triplet<lcg_complex> triplt_cd;

void read(std::string filePath, int *pN, int *pnz, lcg_complex **cooVal, 
	int **cooRowIdx, int **cooColIdx, lcg_complex **b)
{
	std::ifstream in(filePath, std::ios::binary);

	in.read((char*)pN, sizeof(int));
	in.read((char*)pnz, sizeof(int));

	*cooVal = new lcg_complex[*pnz]{};
	*cooRowIdx = new int[*pnz]{};
	*cooColIdx = new int[*pnz]{};
	*b = new lcg_complex[*pN]{};

	std::complex<double> std_c;
	for (int i = 0; i < *pnz; ++i)
	{
		in.read((char*)&(*cooRowIdx)[i], sizeof(int));
		in.read((char*)&(*cooColIdx)[i], sizeof(int));
		in.read((char*)&std_c, sizeof(std_c));
		(*cooVal)[i].real(std_c.real());
		(*cooVal)[i].imag(std_c.imag());
	}

	for (int i = 0; i < *pN; i++)
	{
		in.read((char*)&std_c, sizeof(std_c));
		(*b)[i].real(std_c.real());
		(*b)[i].imag(std_c.imag());
	}
    return;
}

void readAnswer(std::string filePath, int *pN, lcg_complex **x)
{
	std::ifstream in(filePath, std::ios::binary);

	in.read((char*)pN, sizeof(int));

	*x = new lcg_complex[*pN]{};

	std::complex<double> std_c;
	for (size_t i = 0; i < *pN; i++)
	{
		in.read((char*)&std_c, sizeof(std_c));
		(*x)[i].real(std_c.real());
		(*x)[i].imag(std_c.imag());
	}
    return;
}

lcg_float max_diff(const Eigen::VectorXcd &a, const Eigen::VectorXcd &b)
{
	lcg_float max = -1;
	std::complex<lcg_float> t;
	for (int i = 0; i < a.size(); i++)
	{
		t = a[i] - b[i];
		max = lcg_max(t.real()*t.real() + t.imag()*t.imag(), max);
	}
	return max;
}

class TESTFUNC : public CLCG_EIGEN_Solver
{
public:
	TESTFUNC(int n);
	~TESTFUNC();

	void set_kernel(int *row_id, int *col_id, lcg_complex *val, int nz_size);
	void set_p();

	//定义共轭梯度中Ax的算法
	void AxProduct(const Eigen::VectorXcd &x, Eigen::VectorXcd &prod_Ax, 
		lcg_matrix_e layout, clcg_complex_e conjugate)
	{
		if (conjugate == Conjugate) prod_Ax = kernel.conjugate() * x;
		else prod_Ax = kernel * x;
		return;
	}

	void MxProduct(const Eigen::VectorXcd &x, Eigen::VectorXcd &prod_Mx, 
		lcg_matrix_e layout, clcg_complex_e conjugate)
	{
		prod_Mx = P.cwiseProduct(x);
		return;
	}

private:
	spmat_cd kernel;
	Eigen::VectorXcd P;
	int n_size;
};

TESTFUNC::TESTFUNC(int n)
{
	n_size = n;
	kernel.resize(n_size, n_size);
	kernel.setZero();
	P.resize(n_size);
}

TESTFUNC::~TESTFUNC()
{
	kernel.resize(0, 0);
}

void TESTFUNC::set_kernel(int *row_id, int *col_id, lcg_complex *val, int nz_size)
{
	std::vector<triplt_cd> val_triplt;
	for (size_t i = 0; i < nz_size; i++)
	{
		val_triplt.push_back(triplt_cd(row_id[i], col_id[i], val[i]));
	}

	kernel.setFromTriplets(val_triplt.begin(), val_triplt.end());
	return;
}

void TESTFUNC::set_p()
{
	for (size_t i = 0; i < n_size; i++)
	{
		P[i] = 1.0/kernel.coeff(i, i);
	}
	return;
}

int main(int argc, char const *argv[])
{
	std::string inputPath = "data/case_10K_cA";
	std::string answerPath = "data/case_10K_cB";

	int N;
	int nz;
	lcg_complex *A;
	int *rowIdxA;
	int *colIdxA;
	lcg_complex *b;
	read(inputPath, &N, &nz, &A, &rowIdxA, &colIdxA, &b);

	lcg_complex *ans_x;
	readAnswer(answerPath, &N, &ans_x);

	std::clog << "N = " << N << std::endl;
	std::clog << "nz = " << nz << std::endl;

	TESTFUNC test(N);
	test.set_kernel(rowIdxA, colIdxA, A, nz);
	test.set_p();

	Eigen::VectorXcd B, ANS;
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

	// 声明一组解
	Eigen::VectorXcd m = Eigen::VectorXcd::Constant(N, std::complex<double>(0.0, 0.0));

	test.Minimize(m, B, CLCG_BICG);
	std::clog << "maximal difference: " << max_diff(ANS, m) << std::endl << std::endl;

	m.setZero();
	test.Minimize(m, B, CLCG_BICG_SYM);
	std::clog << "maximal difference: " << max_diff(ANS, m) << std::endl << std::endl;

	m.setZero();
	test.Minimize(m, B, CLCG_CGS);
	std::clog << "maximal difference: " << max_diff(ANS, m) << std::endl << std::endl;

	m.setZero();
	test.Minimize(m, B, CLCG_TFQMR);
	std::clog << "maximal difference: " << max_diff(ANS, m) << std::endl << std::endl;

	m.setZero();
	test.MinimizePreconditioned(m, B, CLCG_PCG);
	std::clog << "maximal difference: " << max_diff(ANS, m) << std::endl << std::endl;

	m.setZero();
	test.MinimizePreconditioned(m, B, CLCG_PBICG);
	std::clog << "maximal difference: " << max_diff(ANS, m) << std::endl << std::endl;

	B.resize(0);
	ANS.resize(0);
	m.resize(0);

	delete[] A;
	delete[] rowIdxA;
	delete[] colIdxA;
	delete[] b;
	delete[] ans_x;
	return 0;
}