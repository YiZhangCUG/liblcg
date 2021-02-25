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
		for (int i = 0; i < M; i++)
		{
			tmp_arr[i] = 0.0;
			for (int j = 0; j < num; j++)
			{
				tmp_arr[i] += kernel[i][j] * a[j];
			}
		}

		for (int j = 0; j < num; j++)
		{
			b[j] = 0.0;
			for (int i = 0; i < M; i++)
			{
				b[j] += kernel[i][j] * tmp_arr[i];
			}
		}
		return;
	}

private:
	// 普通二维数组做核矩阵
	lcg_float **kernel;
	// 中间结果数组
	lcg_float *tmp_arr;
};

TESTFUNC::TESTFUNC()
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
}

TESTFUNC::~TESTFUNC()
{
	for (int i = 0; i < M; i++)
	{
		delete[] kernel[i];
	}
	delete[] kernel;
	delete[] tmp_arr;
}

void TESTFUNC::cal_partb(lcg_float *B, const lcg_float *x)
{
	for (int i = 0; i < M; i++)
	{
		tmp_arr[i] = 0.0;
		for (int j = 0; j < N; j++)
		{
			tmp_arr[i] += kernel[i][j]*x[j];
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
}

int main(int argc, char const *argv[])
{
	// 生成一组正演解
	double *fm = new double [N];
	for (int i = 0; i < N; i++)
	{
		fm[i] = random_double(1, 2);
	}

	TESTFUNC test;

	// 计算共轭梯度B项
	double *B = new double [N];
	test.cal_partb(B, fm);

	/********************准备工作完成************************/
	lcg_para self_para = lcg_default_parameters();
	self_para.max_iterations = 1000;
	self_para.epsilon = 1e-3;
	self_para.abs_diff = 1;
	test.set_lcg_parameter(self_para);

	// 声明一组解
	double *m = new double [N];
	for (int i = 0; i < N; i++)
		m[i] = 0.0;

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

	test.Minimize(m, B, N, LCG_CG);

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	test.Minimize(m, B, N, LCG_PCG, p);

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	test.Minimize(m, B, N, LCG_CGS);

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	test.Minimize(m, B, N, LCG_BICGSTAB);

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	test.Minimize(m, B, N, LCG_BICGSTAB2);

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	test.MinimizeConstrained(m, B, low, hig, N, LCG_PG);

	for (int i = 0; i < N; i++)
		m[i] = 0.0;

	test.MinimizeConstrained(m, B, low, hig, N, LCG_SPG);

	delete[] fm;
	delete[] B;
	delete[] m;
	delete[] p;
	delete[] low;
	delete[] hig;
	return 0;
}