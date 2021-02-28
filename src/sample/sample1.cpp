#include "../lib/lcg.h"
#include "ctime"
#include "random"
#include "iostream"

#define M 100
#define N 80

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

// 计算核矩阵乘向量的乘积
void CalAx(void* instance, const lcg_float* x, lcg_float* prod_Ax, const int n_s)
{
	lcg_matvec(kernel, x, tmp_arr, M, n_s, Normal);
	lcg_matvec(kernel, tmp_arr, prod_Ax, M, n_s, Transpose);
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

	srand(time(0));

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			kernel[i][j] = random_lcg_float(-1.0, 1.0);
		}
	}

	// 添加一些大数
	int tmp_id, tmp_size;
	lcg_float tmp_val;
	for (int i = 0; i < M; i++)
	{
		tmp_size = random_int(25, 35);
		for (int j = 0; j < tmp_size; j++)
		{
			tmp_id = random_int(0, N);
			tmp_val = random_lcg_float(-10, 10);

			kernel[i][tmp_id] = tmp_val;
		}
	}

	// 生成一组正演解
	lcg_float *fm = lcg_malloc(N);
	for (int i = 0; i < N; i++)
	{
		fm[i] = random_lcg_float(1, 2);
	}

	// 计算共轭梯度B项
	lcg_float *B = lcg_malloc(N);
	lcg_matvec(kernel, fm, tmp_arr, M, N, Normal);
	lcg_matvec(kernel, tmp_arr, B, M, N, Transpose);

	/********************准备工作完成************************/
	lcg_para self_para = lcg_default_parameters();
	self_para.max_iterations = 1000;
	self_para.epsilon = 1e-3;
	self_para.abs_diff = 1;

	// 声明一组解
	lcg_float *m = lcg_malloc(N);
	lcg_vecset(m, 0.0, N);

	// 声明一组预优因子
	lcg_float *p = lcg_malloc(N);
	lcg_vecset(p, 1.0, N);

	// 约束解的范围
	lcg_float *low = lcg_malloc(N);
	lcg_float *hig = lcg_malloc(N);
	lcg_vecset(low, 1.0, N);
	lcg_vecset(hig, 2.0, N);

	int ret;

	std::clog << "solver: cg" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_CG);
	std::clog << std::endl << lcg_error_str(ret) << std::endl;
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: pcg" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_PCG, p);
	std::clog << std::endl << lcg_error_str(ret) << std::endl;
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: cgs" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_CGS);
	std::clog << std::endl << lcg_error_str(ret) << std::endl;
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: bicgstab" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_BICGSTAB);
	std::clog << std::endl << lcg_error_str(ret) << std::endl;
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: bicgstab2" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_BICGSTAB2);
	std::clog << std::endl << lcg_error_str(ret) << std::endl;
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: pg" << std::endl;
	ret = lcg_solver_constrained(CalAx, Prog, m, B, low, hig, N, &self_para, NULL, LCG_PG);
	std::cerr << std::endl << lcg_error_str(ret) << std::endl;
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	std::clog << "solver: spg" << std::endl;
	ret = lcg_solver_constrained(CalAx, Prog, m, B, low, hig, N, &self_para, NULL, LCG_SPG);
	std::cerr << std::endl << lcg_error_str(ret) << std::endl;
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