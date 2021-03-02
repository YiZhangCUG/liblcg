#include "../lib/lcg.h"
#include "random"

#define M 1000
#define N 800

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
		lcg_matvec(kernel, a, tmp_arr, M, num, Normal);
		lcg_matvec(kernel, tmp_arr, b, M, num, Transpose);
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
	kernel = lcg_malloc(M, N);
	tmp_arr = lcg_malloc(M);

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
	lcg_free(kernel, M);
	lcg_free(tmp_arr);
}

void TESTFUNC::cal_partb(lcg_float *B, const lcg_float *x)
{
	lcg_matvec(kernel, x, tmp_arr, M, N, Normal);
	lcg_matvec(kernel, tmp_arr, B, M, N, Transpose);
}

int main(int argc, char const *argv[])
{
	// 生成一组正演解
	double *fm = lcg_malloc(N);
	for (int i = 0; i < N; i++)
	{
		fm[i] = random_double(1, 2);
	}

	TESTFUNC test;

	// 计算共轭梯度B项
	double *B = lcg_malloc(N);
	test.cal_partb(B, fm);

	/********************准备工作完成************************/
	lcg_para self_para = lcg_default_parameters();
	self_para.max_iterations = 1000;
	self_para.epsilon = 1e-3;
	self_para.abs_diff = 1;
	test.set_lcg_parameter(self_para);

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

	test.Minimize(m, B, N, LCG_CG);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, 0.0, N);
	test.Minimize(m, B, N, LCG_PCG, p);
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
	lcg_free(p);
	lcg_free(low);
	lcg_free(hig);
	return 0;
}