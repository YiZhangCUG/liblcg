#include "../lib/clcg.h"
#include "ctime"
#include "random"
#include "iostream"
#include "iomanip"

#define N 25

//返回范围内的随机浮点值 注意调取函数之前要调用srand(time(0));
lcg_float random_lcg_float(lcg_float L,lcg_float T)
{
	return (T-L)*rand()*1.0/RAND_MAX + L;
}

//返回范围内的随机整数 注意调取函数之前要调用srand(time(0));
int random_int(int small, int big)
{
	return (rand() % (big - small))+ small;
}

lcg_float max_diff(const lcg_complex *a, const lcg_complex *b, int size)
{
	lcg_float max = -1;
	for (int i = 0; i < size; i++)
	{
		max = lcg_max((a[i] - b[i]).module(), max);
	}
	return max;
}

// 普通二维数组做核矩阵
lcg_complex **kernel;

// 计算核矩阵乘向量的乘积
void CalAx(void *instance, const lcg_complex *x, lcg_complex *prod_Ax, 
	const int x_size, matrix_layout_e layout, complex_conjugate_e conjugate)
{
	lcg_matvec(kernel, x, prod_Ax, N, x_size, layout, conjugate);
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

	kernel = lcg_malloc_complex(N, N);

	for (int i = 0; i < N; i++)
	{
		for (int j = i; j < N; j++)
		{
			kernel[i][j].rel = random_lcg_float(-1.0, 1.0);
			kernel[i][j].img = random_lcg_float(-1.0, 1.0);
			kernel[j][i] = kernel[i][j];
		}
	}

	// 添加一些大数
	int tmp_id, tmp_size;
	for (int i = 0; i < N; i++)
	{
		tmp_size = random_int(5, 10);
		for (int j = 0; j < tmp_size; j++)
		{
			tmp_id = random_int(0, N);
			kernel[i][tmp_id].rel = random_lcg_float(-10, 10);
			kernel[i][tmp_id].img = random_lcg_float(-10, 10);
			kernel[tmp_id][i] = kernel[i][tmp_id];
		}
	}

	// 生成一组正演解
	lcg_complex *fm = lcg_malloc_complex(N);
	for (int i = 0; i < N; i++)
	{
		fm[i].rel = random_lcg_float(1, 2);
		fm[i].img = random_lcg_float(1, 2);
	}

	// 计算共轭梯度B项
	lcg_complex *B = lcg_malloc_complex(N);
	lcg_matvec(kernel, fm, B, N, N, Normal, NonConjugate);

	/********************准备工作完成************************/
	clcg_para self_para = clcg_default_parameters();
	self_para.max_iterations = 1000;

	// 声明一组解
	lcg_complex *m = lcg_malloc_complex(N);
	lcg_vecset(m, lcg_complex(0.0, 0.0), N);

	int ret;

	std::clog << "solver: bicg" << std::endl;
	ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_BICG);
	std::clog << std::endl << clcg_error_str(ret) << std::endl;
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, lcg_complex(0.0, 0.0), N);
	std::clog << "solver: bicg-symmetric" << std::endl;
	ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_BICG_SYM);
	std::clog << std::endl << clcg_error_str(ret) << std::endl;
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, lcg_complex(0.0, 0.0), N);
	std::clog << "solver: cgs" << std::endl;
	ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_CGS);
	std::clog << std::endl << clcg_error_str(ret) << std::endl;
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, lcg_complex(0.0, 0.0), N);
	std::clog << "solver: tfqmr" << std::endl;
	ret = clcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, CLCG_TFQMR);
	std::clog << std::endl << clcg_error_str(ret) << std::endl;
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_free(kernel, N);
	lcg_free(fm);
	lcg_free(B);
	lcg_free(m);
	return 0;
}