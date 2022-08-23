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
#include "../lib/clcg.h"

#define N 100

lcg_float max_diff(const lcg_complex *a, const lcg_complex *b, int size)
{
	lcg_float max = -1;
	lcg_complex t;
	for (int i = 0; i < size; i++)
	{
		t = a[i] - b[i];
		max = lcg_max(clcg_module(&t), max);
	}
	return max;
}

// 普通二维数组做核矩阵
lcg_complex **kernel;

// 计算核矩阵乘向量的乘积
void CalAx(void *instance, const lcg_complex *x, lcg_complex *prod_Ax, 
	const int x_size, lcg_matrix_e layout, clcg_complex_e conjugate)
{
	clcg_matvec(kernel, x, prod_Ax, N, x_size, layout, conjugate);
	return;
}


//定义共轭梯度监控函数
int Prog(void* instance, const lcg_complex* m, const lcg_float converge, 
	const clcg_para* param, const int n_size, const int k)
{
	std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
	return 0;
}

int main(int argc, char const *argv[])
{
	srand(time(0));

	kernel = clcg_malloc(N, N);
	clcg_vecrnd(kernel, lcg_complex(-1.0, -1.0), lcg_complex(1.0, 1.0), N, N);

	// 设置核矩阵为一个对称阵
	for (int i = 0; i < N; i++)
	{
		for (int j = i; j < N; j++)
		{
			kernel[j][i] = kernel[i][j];
		}
	}

	// 生成一组正演解
	lcg_complex *fm = clcg_malloc(N);
	clcg_vecrnd(fm, lcg_complex(1.0, 1.0), lcg_complex(2.0, 2.0), N);

	// 计算共轭梯度B项
	lcg_complex *B = clcg_malloc(N);
	clcg_matvec(kernel, fm, B, N, N, MatNormal, NonConjugate);

	/********************准备工作完成************************/
	clcg_para self_para = clcg_default_parameters();
	self_para.abs_diff = 1;
	self_para.epsilon = 1e-6;

	// 声明一组解
	lcg_complex *m = clcg_malloc(N);
	clcg_vecset(m, lcg_complex(0.0, 0.0), N);

	int ret;

	std::clog << "solver: bicg" << std::endl;
	ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_BICG);
	std::clog << std::endl; clcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	clcg_vecset(m, lcg_complex(0.0, 0.0), N);
	std::clog << "solver: bicg-symmetric" << std::endl;
	ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_BICG_SYM);
	std::clog << std::endl; clcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	clcg_vecset(m, lcg_complex(0.0, 0.0), N);
	std::clog << "solver: cgs" << std::endl;
	ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_CGS);
	std::clog << std::endl; clcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	clcg_vecset(m, lcg_complex(0.0, 0.0), N);
	std::clog << "solver: bicgstab" << std::endl;
	ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_BICGSTAB);
	std::clog << std::endl; clcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	clcg_vecset(m, lcg_complex(0.0, 0.0), N);
	std::clog << "solver: tfqmr" << std::endl;
	ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_TFQMR);
	std::clog << std::endl; clcg_error_str(ret);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	clcg_free(kernel, N);
	clcg_free(fm);
	clcg_free(B);
	clcg_free(m);
	return 0;
}