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
#include "random"
#include "../lib/solver.h"

#define M 1000
#define N 800

lcg_float max_diff(const lcg_float *a, const lcg_float *b, int size)
{
	lcg_float max = -1;
	for (int i = 0; i < size; i++)
	{
		max = lcg_max(sqrt((a[i] - b[i])*(a[i] - b[i])), max);
	}
	return max;
}

class TESTFUNC : public LCG_Solver
{
public:
	TESTFUNC();
	~TESTFUNC();

	// 计算共轭梯度的B项
	void cal_partb(lcg_float *B, const lcg_float *x);

	//定义共轭梯度中Ax的算法
	void AxProduct(const lcg_float* a, lcg_float* b, const int num)
	{
		lcg_matvec(kernel, a, tmp_arr, M, num, MatNormal);
		lcg_matvec(kernel, tmp_arr, b, M, num, MatTranspose);
		return;
	}

	void MxProduct(const lcg_float* a, lcg_float* b, const int num)
	{
		for (size_t i = 0; i < num; i++)
		{
			b[i] = p[i]*a[i];
		}
		return;
	}

private:
	// 普通二维数组做核矩阵
	lcg_float **kernel;
	// 中间结果数组
	lcg_float *tmp_arr;
	// 预优矩阵
	lcg_float *p;
};

TESTFUNC::TESTFUNC()
{
	kernel = lcg_malloc(M, N);
	tmp_arr = lcg_malloc(M);
	p = lcg_malloc(N);

	lcg_vecrnd(kernel, -1.0, 1.0, M, N);
	lcg_vecset(p, 1.0, N);

	lcg_float diag;
	for (size_t i = 0; i < N; i++)
	{
		diag = 0.0;
		for (size_t j = 0; j < M; j++)
		{
			diag += kernel[j][i]*kernel[j][i];
		}
		p[i] = 1.0/diag;
	}
}

TESTFUNC::~TESTFUNC()
{
	lcg_free(kernel, M);
	lcg_free(tmp_arr);
	lcg_free(p);
}

void TESTFUNC::cal_partb(lcg_float *B, const lcg_float *x)
{
	lcg_matvec(kernel, x, tmp_arr, M, N, MatNormal);
	lcg_matvec(kernel, tmp_arr, B, M, N, MatTranspose);
}

int main(int argc, char const *argv[])
{
	// 生成一组正演解
	double *fm = lcg_malloc(N);
	lcg_vecrnd(fm, 1.0, 2.0, N);

	TESTFUNC test;

	// 计算共轭梯度B项
	double *B = lcg_malloc(N);
	test.cal_partb(B, fm);

	/********************准备工作完成************************/
	lcg_para self_para = lcg_default_parameters();
	self_para.epsilon = 1e-3;
	self_para.abs_diff = 1;
	test.set_lcg_parameter(self_para);

	// 声明一组解
	lcg_float *m = lcg_malloc(N);
	lcg_vecset(m, 0.0, N);

	// 约束解的范围
	lcg_float *low = lcg_malloc(N);
	lcg_float *hig = lcg_malloc(N);
	lcg_vecset(low, 1.0, N);
	lcg_vecset(hig, 2.0, N);

	test.Minimize(m, B, N, LCG_CG);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	test.MinimizePreconditioned(m, B, N);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	test.Minimize(m, B, N, LCG_CGS);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	test.Minimize(m, B, N, LCG_BICGSTAB);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	test.Minimize(m, B, N, LCG_BICGSTAB2);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	test.MinimizeConstrained(m, B, low, hig, N, LCG_PG);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	test.MinimizeConstrained(m, B, low, hig, N, LCG_SPG);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_free(fm);
	lcg_free(B);
	lcg_free(m);
	lcg_free(low);
	lcg_free(hig);
	return 0;
}