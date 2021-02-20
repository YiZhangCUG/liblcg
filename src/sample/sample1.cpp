#include "../lib/lcg.h"
#include "ctime"
#include "random"
#include "iostream"

#define M 100
#define N 80

//返回范围内的随机浮点值 注意调取函数之前要调用srand(time(0));
double random_double(double L,double T)
{
	return (T-L)*rand()*1.0/RAND_MAX + L;
}

//返回范围内的随机整数 注意调取函数之前要调用srand(time(0));
int random_int(int small, int big)
{
	return (rand() % (big - small))+ small;
}

// 普通二维数组做核矩阵
double **kernel;
// 中间结果数组
double *tmp_arr;

// 计算核矩阵乘向量的乘积
void CalAx(void* instance, const lcg_float* x, lcg_float* prod_Ax, const int n_s)
{
	for (int i = 0; i < M; i++)
	{
		tmp_arr[i] = 0.0;
		for (int j = 0; j < n_s; j++)
		{
			tmp_arr[i] += kernel[i][j] * x[j];
		}
	}

	for (int j = 0; j < n_s; j++)
	{
		prod_Ax[j] = 0.0;
		for (int i = 0; i < M; i++)
		{
			prod_Ax[j] += kernel[i][j] * tmp_arr[i];
		}
	}
	return;
}

//定义共轭梯度监控函数
int Prog(void* instance, const lcg_float* m, const lcg_float converge, const lcg_para* param, const int n_s, const int k)
{
	std::clog << "Iteration-times: " << k << "\tconvergence: " << converge << std::endl;
#ifdef __linux__
	if (converge > param->epsilon) std::clog << "\033[1A\033[K";
#elif defined __APPLE__
	if (converge > param->epsilon) std::clog << "\033[1A\033[K";
#endif
	return 0;
}

int main(int argc, char const *argv[])
{
	kernel = new double *[M];
	for (int i = 0; i < M; i++)
	{
		kernel[i] = new double [N];
	}
	tmp_arr = new double [M];

	srand(time(0));

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			kernel[i][j] = random_double(-1.0, 1.0);
		}
	}

	// 添加一些大数
	int tmp_id, tmp_size;
	double tmp_val;
	for (int i = 0; i < M; i++)
	{
		tmp_size = random_int(25, 35);
		for (int j = 0; j < tmp_size; j++)
		{
			tmp_id = random_int(0, N);
			tmp_val = random_double(-10, 10);

			kernel[i][tmp_id] = tmp_val;
		}
	}

	// 生成一组正演解
	double *fm = new double [N];
	for (int i = 0; i < N; i++)
	{
		fm[i] = random_double(1, 2);
	}

	// 计算共轭梯度B项
	double *B = new double [N];
	for (int i = 0; i < M; i++)
	{
		tmp_arr[i] = 0.0;
		for (int j = 0; j < N; j++)
		{
			tmp_arr[i] += kernel[i][j]*fm[j];
		}
	}

	for (int j = 0; j < N; j++)
	{
		B[j] = 0.0;
		for (int i = 0; i < M; i++)
		{
			B[j] += kernel[i][j]*tmp_arr[i];
		}
	}

	/********************准备工作完成************************/
	lcg_para self_para = lcg_default_parameters();
	self_para.max_iterations = 1000;
	self_para.epsilon = 1e-3;
	self_para.abs_diff = 1;

	// 声明一组解
	double *m = new double [N];
	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	// 声明一组预优因子
	double *p = new double [N];
	for (int i = 0; i < N; i++)
		p[i] = 1.0;

	// 约束解的范围
	double *low = new double [N];
	double *hig = new double [N];
	for (int i = 0; i < N; i++)
	{
		low[i] = 1.0;
		hig[i] = 2.0;
	}

	int ret;

	std::clog << "solver: cg" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_CG);
	std::clog << lcg_error_str(ret) << std::endl;

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	std::clog << "solver: pcg" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_PCG, p);
	std::clog << lcg_error_str(ret) << std::endl;

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	std::clog << "solver: cgs" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_CGS);
	std::clog << lcg_error_str(ret) << std::endl;

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	std::clog << "solver: bicgstab" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_BICGSTAB);
	std::clog << lcg_error_str(ret) << std::endl;

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	std::clog << "solver: bicgstab2" << std::endl;
	ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_BICGSTAB2);
	std::clog << lcg_error_str(ret) << std::endl;

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	std::clog << "solver: pg" << std::endl;
	ret = lcg_solver_constrained(CalAx, Prog, m, B, low, hig, N, &self_para, NULL, LCG_PG);
	std::cerr << lcg_error_str(ret) << std::endl;

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	std::clog << "solver: spg" << std::endl;
	ret = lcg_solver_constrained(CalAx, Prog, m, B, low, hig, N, &self_para, NULL, LCG_SPG);
	std::cerr << lcg_error_str(ret) << std::endl;

	delete[] kernel;
	delete[] tmp_arr;
	delete[] fm;
	delete[] B;
	delete[] m;
	delete[] p;
	delete[] low;
	delete[] hig;
	return 0;
}