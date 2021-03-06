#include "../lib/clcg.h"
#include "ctime"
#include "random"
#include "iostream"
#include "iomanip"

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

lcg_float max_diff(const lcg_complex *a, const lcg_complex *b, int size)
{
	lcg_float max = -1;
	for (int i = 0; i < size; i++)
	{
		max = lcg_max((a[i] - b[i]).module(), max);
	}
	return max;
}

class TESTFUNC : public CLCG_Solver
{
public:
	TESTFUNC();
	~TESTFUNC();

	// 计算共轭梯度的B项
	void cal_partb(lcg_complex *B, const lcg_complex *x);

	//定义共轭梯度中Ax的算法
	void AxProduct(const lcg_complex *x, lcg_complex *prod_Ax, const int x_size, 
		matrix_layout_e layout, complex_conjugate_e conjugate)
	{
		lcg_matvec(kernel, x, tmp_arr, M, x_size, Normal, conjugate);
		lcg_matvec(kernel, tmp_arr, prod_Ax, M, x_size, Transpose, conjugate);
		return;
	}

private:
	// 普通二维数组做核矩阵
	lcg_complex **kernel;
	// 中间结果数组
	lcg_complex *tmp_arr;
};

TESTFUNC::TESTFUNC()
{
	kernel = lcg_malloc_complex(M, N);
	tmp_arr = lcg_malloc_complex(M);

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			kernel[i][j].rel = 0.0;//random_lcg_float(-1.0, 1.0);
			kernel[i][j].img = 0.0;//random_lcg_float(-1.0, 1.0);
		}
	}

	// 添加一些大数
	int tmp_id, tmp_size;
	for (int i = 0; i < M; i++)
	{
		tmp_size = random_int(25, 35);
		for (int j = 0; j < tmp_size; j++)
		{
			tmp_id = random_int(0, N);
			kernel[i][tmp_id].rel = random_lcg_float(-10, 10);
			kernel[i][tmp_id].img = random_lcg_float(-10, 10);
		}
	}
}

TESTFUNC::~TESTFUNC()
{
	lcg_free(kernel, M);
	lcg_free(tmp_arr);
}

void TESTFUNC::cal_partb(lcg_complex *B, const lcg_complex *x)
{
	lcg_matvec(kernel, x, tmp_arr, M, N, Normal);
	lcg_matvec(kernel, tmp_arr, B, M, N, Transpose);
	return;
}

int main(int argc, char const *argv[])
{
	srand(time(0));

	// 声明一组解
	lcg_complex *fm = lcg_malloc_complex(N);
	for (int i = 0; i < N; i++)
	{
		fm[i].rel = random_lcg_float(1, 2);
		fm[i].img = random_lcg_float(1, 2);
	}

	TESTFUNC test;

	// 计算共轭梯度B项
	lcg_complex *B = lcg_malloc_complex(N);
	test.cal_partb(B, fm);

	/********************准备工作完成************************/
	clcg_para self_para = clcg_default_parameters();
	self_para.max_iterations = 1000;
	self_para.epsilon = 1e-3;
	self_para.abs_diff = 1;
	test.set_clcg_parameter(self_para);

	// 声明一组解
	lcg_complex *m = lcg_malloc_complex(N);
	lcg_vecset(m, lcg_complex(0.0, 0.0), N);

	test.Minimize(m, B, N, CLCG_BICG);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, lcg_complex(0.0, 0.0), N);
	test.Minimize(m, B, N, CLCG_BICG_SYM);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, lcg_complex(0.0, 0.0), N);
	test.Minimize(m, B, N, CLCG_CGS);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_vecset(m, lcg_complex(0.0, 0.0), N);
	test.Minimize(m, B, N, CLCG_TFQMR);
	std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

	lcg_free(fm);
	lcg_free(B);
	lcg_free(m);
	return 0;
}