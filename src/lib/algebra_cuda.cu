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

#include "algebra_cuda.h"


__global__ void lcg_set2box_cuda_device(const lcg_float *low, const lcg_float *hig, lcg_float *a, 
    int n, bool low_bound, bool hig_bound)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		if (hig_bound && a[i] >= hig[i]) a[i] = hig[i];
		if (!hig_bound && a[i] > hig[i]) a[i] = hig[i];
		if (low_bound && a[i] <= low[i]) a[i] = low[i];
		if (!low_bound && a[i] < low[i]) a[i] = low[i];
	}
	return;
}

__global__ void lcg_smDcsr_get_diagonal_device(const int *A_ptr, const int *A_col, const lcg_float *A_val, const int A_len, lcg_float *A_diag)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < A_len)
	{
		const int num_non0_row = A_ptr[i + 1] - A_ptr[i];

		for (int j = 0; j < num_non0_row; j++)
		{
			if (A_col[j + A_ptr[i]] == i)
			{
				A_diag[i] = A_val[j + A_ptr[i]];
				break;
			}
		}
	}
	return;
}

__global__ void lcg_vecMvecD_element_wise_device(const lcg_float *a, const lcg_float *b, lcg_float *c, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		c[i] = a[i] * b[i];
	}
	return;
}

__global__ void lcg_vecDvecD_element_wise_device(const lcg_float *a, const lcg_float *b, lcg_float *c, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		c[i] = a[i] / b[i];
	}
	return;
}

void lcg_set2box_cuda(const lcg_float *low, const lcg_float *hig, lcg_float *a, 
    int n, bool low_bound, bool hig_bound)
{
	int blockSize = 1024;
	int numBlocks = (n+ blockSize - 1) / blockSize;
	lcg_set2box_cuda_device<<<numBlocks, blockSize>>>(low, hig, a, n, low_bound, hig_bound);
	return;
}

void lcg_smDcsr_get_diagonal(const int *A_ptr, const int *A_col, const lcg_float *A_val, const int A_len, lcg_float *A_diag, int bk_size)
{
	int blockSize = bk_size;
	int numBlocks = (A_len+ blockSize - 1) / blockSize;
	lcg_smDcsr_get_diagonal_device<<<numBlocks, blockSize>>>(A_ptr, A_col, A_val, A_len, A_diag);
	return;
}

void lcg_vecMvecD_element_wise(const lcg_float *a, const lcg_float *b, lcg_float *c, int n, int bk_size)
{
	int blockSize = bk_size;
	int numBlocks = (n + blockSize - 1) / blockSize;
	lcg_vecMvecD_element_wise_device<<<numBlocks, blockSize>>>(a, b, c, n);
	return;
}

void lcg_vecDvecD_element_wise(const lcg_float *a, const lcg_float *b, lcg_float *c, int n, int bk_size)
{
	int blockSize = bk_size;
	int numBlocks = (n + blockSize - 1) / blockSize;
	lcg_vecDvecD_element_wise_device<<<numBlocks, blockSize>>>(a, b, c, n);
	return;
}