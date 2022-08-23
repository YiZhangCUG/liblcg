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

#include "cmath"
#include "iostream"
#include "../lib/lcg.h"

#define M 100
#define N 80

lcg_float max_diff(const lcg_float *a, const lcg_float *b, int size)
{
	lcg_float max = -1;
	for (int i = 0; i < size; i++)
	{
		max = lcg_max(sqrt((a[i] - b[i])*(a[i] - b[i])), max);
	}
	return max;
}

// 普通二维数组做核矩阵
lcg_float **kernel;
// 中间结果数组
lcg_float *tmp_arr;
// 预优矩阵
lcg_float *p;

// 计算核矩阵乘向量的乘积
void CalAx(void* instance, const lcg_float* x, lcg_float* prod_Ax, const int n_s)
{
	lcg_matvec(kernel, x, tmp_arr, M, n_s, MatNormal);
	lcg_matvec(kernel, tmp_arr, prod_Ax, M, n_s, MatTranspose);
	return;
}

void CalMx(void* instance, const lcg_float* x, lcg_float* prod_Mx, const int n_s)
{
	for (size_t i = 0; i < n_s; i++)
	{
		prod_Mx[i] = p[i]*x[i];
	}
	return;
}

//定义共轭梯度监控函数
int Prog(void* instance, const lcg_float* m, const lcg_float converge, const lcg_para* param, const int n_s, const int k)
{
	std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
	return 0;
}

int main(int argc, char const *argv[])
{
	kernel = lcg_malloc(M, N);
	tmp_arr = lcg_malloc(M);
	p = lcg_malloc(N);

	lcg_vecrnd(kernel, -1.0, 1.0, M, N);

	// 生成一组正演解
	lcg_float *fm = lcg_malloc(N);
	lcg_vecrnd(fm, 1.0, 2.0, N);

	// 计算共轭梯度B项
	lcg_float *B = lcg_malloc(N);
	lcg_matvec(kernel, fm, tmp_arr, M, N, MatNormal);
	lcg_matvec(kernel, tmp_arr, B, M, N, MatTranspose);

	/********************准备工作完成************************/
	lcg_para self_para = lcg_default_parameters();
	self_para.epsilon = 1e-5;
	self_para.abs_diff = 0;

	// 声明一组解
	lcg_float *m = lcg_malloc(N);
	lcg_vecset(m, 0.0, N);

	// 声明一组预优因子
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

	// 约束解的范围
	lcg_float *low = lcg_malloc(N);
	lcg_float *hig = lcg_malloc(N);
	lcg_vecset(low, 1.0, N);
	lcg_vecset(hig, 2.0, N);

	int ret;

	std::clog << "solver: cg" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_CG);
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: pcg" << std::endl;
	ret = lcg_solver_preconditioned(CalAx, CalMx, Prog, m, B, N, &self_para, NULL, LCG_PCG);
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: cgs" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_CGS);
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: bicgstab" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_BICGSTAB);
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: bicgstab2" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_BICGSTAB2);
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: pg" << std::endl;
	ret = lcg_solver_constrained(CalAx, Prog, m, B, low, hig, N, &self_para, NULL, LCG_PG);
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: spg" << std::endl;
	ret = lcg_solver_constrained(CalAx, Prog, m, B, low, hig, N, &self_para, NULL, LCG_SPG);
	std::clog << std::endl; lcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_free(kernel, M);
	lcg_free(tmp_arr);
	lcg_free(fm);
	lcg_free(B);
	lcg_free(m);
	lcg_free(p);
	lcg_free(low);
	lcg_free(hig);
	return 0;
}