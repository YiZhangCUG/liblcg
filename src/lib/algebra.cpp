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

#include "ctime"
#include "random"

#include "algebra.h"

#ifdef LibLCG_OPENMP
#include "omp.h"
#endif

lcg_float lcg_abs(lcg_float a)
{
	if (a >= 0.0) return a;
	return -1.0*a;
}

lcg_float lcg_max(lcg_float a, lcg_float b)
{
	if (a >= b) return a;
	return b;
}

lcg_float lcg_min(lcg_float a, lcg_float b)
{
	if (a <= b) return a;
	return b;
}

lcg_float lcg_set2box(lcg_float low, lcg_float hig, lcg_float a, 
	bool low_bound, bool hig_bound)
{
	if (hig_bound && a >= hig) return hig;
	if (!hig_bound && a >= hig) return (hig - 1e-16);
	if (low_bound && a <= low) return low;
	if (!low_bound && a <= low) return (low + 1e-16);
	return a;
}

lcg_float* lcg_malloc(int n)
{
	lcg_float* x = new lcg_float [n];
	return x;
}

lcg_float** lcg_malloc(int m, int n)
{
	lcg_float **x = new lcg_float* [m];
	for (int i = 0; i < m; i++)
	{
		x[i] = new lcg_float [n];
	}
	return x;
}

void lcg_free(lcg_float* x)
{
	if (x != nullptr)
	{
		delete[] x;
		x = nullptr;
	}
	return;
}

void lcg_free(lcg_float **x, int m)
{
	if (x != nullptr)
	{
		for (int i = 0; i < m; i++)
		{
			delete[] x[i];
		}
		delete[] x;
		x = nullptr;
	}
	return;
}

void lcg_vecset(lcg_float *a, lcg_float b, int size)
{
	for (int i = 0; i < size; i++)
	{
		a[i] = b;
	}
	return;
}

void lcg_vecset(lcg_float **a, lcg_float b, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            a[i][j] = b;
        }
    }
    return;
}

void lcg_vecrnd(lcg_float *a, lcg_float l, lcg_float h, int size)
{
	srand(time(nullptr));
	for (int i = 0; i < size; i++)
	{
		a[i] = (h-l)*rand()*1.0/RAND_MAX + l;
	}
	return;
}

void lcg_vecrnd(lcg_float **a, lcg_float l, lcg_float h, int m, int n)
{
	srand(time(nullptr));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a[i][j] = (h-l)*rand()*1.0/RAND_MAX + l;	
		}
	}
	return;
}

double lcg_squaredl2norm(lcg_float *a, int n)
{
	lcg_float sum = 0;
	for (size_t i = 0; i < n; i++)
	{
		sum += a[i]*a[i];
	}
	return sum;
}

void lcg_dot(lcg_float &ret, const lcg_float *a, 
	const lcg_float *b, int size)
{
	ret = 0.0;
	for (int i = 0; i < size; i++)
	{
		ret += a[i]*b[i];
	}
	return;
}

void lcg_matvec(lcg_float **A, const lcg_float *x, lcg_float *Ax, 
	int m_size, int n_size, lcg_matrix_e layout)
{
	int i, j;
	if (layout == MatNormal)
	{
#pragma omp parallel for private (i, j) schedule(guided)
		for (i = 0; i < m_size; i++)
		{
			Ax[i] = 0.0;
			for (j = 0; j < n_size; j++)
			{
				Ax[i] += A[i][j]*x[j];
			}
		}
		return;
	}

#pragma omp parallel for private (i, j) schedule(guided)
	for (j = 0; j < n_size; j++)
	{
		Ax[j] = 0.0;
		for (i = 0; i < m_size; i++)
		{
			Ax[j] += A[i][j]*x[i];
		}
	}
	return;
}

void lcg_matvec_coo(const int *row, const int *col, const lcg_float *Mat, const lcg_float *V, lcg_float *p, int M, int N, int nz_size, bool pre_position)
{
	if (!pre_position)
	{
		for (size_t i = 0; i < M; i++)
		{
			p[i] = 0.0;
		}

		for (size_t i = 0; i < nz_size; i++)
		{
			p[row[i]] += Mat[i]*V[col[i]];
		}
	}
	else
	{
		for (size_t i = 0; i < N; i++)
		{
			p[i] = 0.0;
		}

		for (size_t i = 0; i < nz_size; i++)
		{
			p[col[i]] += Mat[i]*V[row[i]];
		}
	}
	return;
}