#include "../lib/lcg_cxx.h"
#include "iostream"

using std::cout;
using std::clog;
using std::endl;

class TESTFUNC : public LCG_Solver
{
public:
	TESTFUNC()
	{
		// 测试线性方程组
		// 6.3*x1 + 3.9*x2 + 2.5*x3 = -2.37
		// 3.9*x1 + 1.2*x2 + 3.1*x3 = 5.82
		// 2.5*x1 + 3.1*x2 + 7.6*x3 = 5.21
		// 目标解 x1=1.2 x2=-3.7 x3=1.8
		// 注意根据共轭梯度法的要求 kernel是一个N阶对称阵
		kernel_[0][0] = 6.3; kernel_[0][1] = 3.9; kernel_[0][2] = 2.5;
		kernel_[1][0] = 3.9; kernel_[1][1] = 1.2; kernel_[1][2] = 3.1;
		kernel_[2][0] = 2.5; kernel_[2][1] = 3.1; kernel_[2][2] = 7.6;
	}

	//定义共轭梯度中Ax的算法
	void AxProduct(const lcg_float* a, lcg_float* b, const int num)
	{
		for (int i = 0; i < num; i++)
		{
			b[i] = 0.0;
			for (int j = 0; j < num; j++)
			{
				b[i] += kernel_[i][j]*a[j];
			}
		}
		return;
	}

private:
	lcg_float kernel_[3][3];
};

int main(int argc, char const *argv[])
{
	lcg_float* m;
	lcg_float* b;
	lcg_float* p;
	// 初始解
	m = lcg_malloc(3);
	m[0] = 0.0; m[1] = 0.0; m[2] = 0.0;
	// 拟合目标值（含有一定的噪声）
	b = lcg_malloc(3);
	b[0] = -2.3723; b[1] = 5.8201; b[2] = 5.2065;
	// 测试预优矩阵 这里只是测试流程 预优矩阵值全为1 并没有什么作用
	p = lcg_malloc(3);
	p[0] = p[1] = p[2] = 1.0;

	TESTFUNC test;
	test.Minimize(m, b, 3, LCG_CG);
	// 输出解
	for (int i = 0; i < 3; i++)
	{
		cout << m[i] << endl;
	}

	m[0] = 0.0; m[1] = 0.0; m[2] = 0.0;
	test.Minimize(m, b, 3, LCG_PCG, p);
	// 输出解
	for (int i = 0; i < 3; i++)
	{
		cout << m[i] << endl;
	}

	lcg_free(m);
	lcg_free(b);
	lcg_free(p);
	return 0;
}