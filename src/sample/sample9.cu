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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

#include "../lib/clcg_cuda.h"

void read(std::string filePath, int *pN, int *pnz, cuDoubleComplex **cooVal,
	int **cooRowIdx, int **cooColIdx, cuDoubleComplex **b)
{
	std::ifstream in(filePath, std::ios::binary);

	in.read((char*)pN, sizeof(int));
	in.read((char*)pnz, sizeof(int));

	*cooVal = new cuDoubleComplex[*pnz]{};
	*cooRowIdx = new int[*pnz]{};
	*cooColIdx = new int[*pnz]{};
	*b = new cuDoubleComplex[*pN]{};

	for (int i = 0; i < *pnz; ++i)
	{
		in.read((char*)&(*cooRowIdx)[i], sizeof(int));
		in.read((char*)&(*cooColIdx)[i], sizeof(int));
		in.read((char*)&(*cooVal)[i], sizeof(cuDoubleComplex));
	}

	in.read((char*)(*b), sizeof(cuDoubleComplex)*(*pN));
    return;
}

void readAnswer(std::string filePath, int *pN, cuDoubleComplex **x)
{
	std::ifstream in(filePath, std::ios::binary);

	in.read((char*)pN, sizeof(int));

	*x = new cuDoubleComplex[*pN]{};

	in.read((char*)(*x), sizeof(cuDoubleComplex)*(*pN));
    return;
}

lcg_float avg_error(cuDoubleComplex *a, cuDoubleComplex *b, int n)
{
	lcg_float avg = 0.0;
	cuDoubleComplex tmp;
	for (size_t i = 0; i < n; i++)
	{
		tmp = clcg_Zdiff(a[i], b[i]);
		avg += (tmp.x*tmp.x + tmp.y*tmp.y);
	}
	return sqrt(avg)/n;
}

// Declare as global variables
cuDoubleComplex one, zero;

void *d_buf;
cusparseSpMatDescr_t smat_A;

int *d_rowIdxA; // COO
int *d_rowPtrA; // CSR
int *d_colIdxA;
cuDoubleComplex *d_A;
cuDoubleComplex *d_B;

void cudaAx(void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
    cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, const int n_size, const int nz_size, 
	cusparseOperation_t oper_t)
{
	one.x = 1.0; one.y = 0.0;
	zero.x = 0.0; zero.y = 0.0;
	// Calculate the product of A*x
	cusparseSpMV(cus_handle, oper_t, &one, smat_A, x, &zero, prod_Ax, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);
    return;
}

int cudaProgress(void* instance, const cuDoubleComplex* m, const lcg_float converge, 
	const clcg_para* param, const int n_size, const int nz_size, const int k)
{
    if (converge <= param->epsilon) {
		std::clog << "Iteration-times: " << k << "\tconvergence: " << converge << std::endl;
	}
	return 0;
}

int main(int argc, char **argv)
{
	std::string inputPath = "data/case_1K_cA";
	std::string answerPath = "data/case_1K_cB";

	int N, nz;
	int *rowIdxA, *colIdxA;
	cuDoubleComplex *A, *b;

	read(inputPath, &N, &nz, &A, &rowIdxA, &colIdxA, &b);

	cuDoubleComplex *ans_x;
	readAnswer(answerPath, &N, &ans_x);

	std::clog << "N = " << N << std::endl;
	std::clog << "nz = " << nz << std::endl;

	// Create handles
	cublasHandle_t cubHandle;
	cusparseHandle_t cusHandle;

	cublasCreate(&cubHandle);
	cusparseCreate(&cusHandle);

	// Allocate GPU memory & copy matrix/vector to device
	cudaMalloc(&d_A, nz * sizeof(cuDoubleComplex));
	cudaMalloc(&d_rowIdxA, nz * sizeof(int));
	cudaMalloc(&d_rowPtrA, (N + 1) * sizeof(int));
	cudaMalloc(&d_colIdxA, nz * sizeof(int));
	cudaMalloc(&d_B, N * sizeof(cuDoubleComplex));

	cudaMemcpy(d_A, A, nz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowIdxA, rowIdxA, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIdxA, colIdxA, nz * sizeof(int), cudaMemcpyHostToDevice);

	// Convert matrix A from COO format to CSR format
	cusparseXcoo2csr(cusHandle, d_rowIdxA, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO);

	// Create sparse matrix
	cusparseCreateCsr(&smat_A, N, N, nz, d_rowPtrA, d_colIdxA, d_A, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

	// This is just used to get bufferSize;
	cusparseDnVecDescr_t dvec_tmp;
	cusparseCreateDnVec(&dvec_tmp, N, d_B, CUDA_C_64F);

	size_t bufferSize_B, bufferSize_B2;

	cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_A,
		dvec_tmp, &zero, dvec_tmp, CUDA_C_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize_B);
	
	cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, &one, smat_A,
		dvec_tmp, &zero, dvec_tmp, CUDA_C_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize_B2);
	
	if (bufferSize_B2 > bufferSize_B) bufferSize_B = bufferSize_B2;
	cudaMalloc(&d_buf, bufferSize_B);

	// Declare an initial solution
    clcg_para self_para = clcg_default_parameters();
	self_para.epsilon = 1e-6;
	self_para.abs_diff = 0;

	int ret;
	cuDoubleComplex *host_m = new cuDoubleComplex[N];

	// Solve with BICG
	for (size_t i = 0; i < N; i++)
	{
		host_m[i].x = 0.0; host_m[i].y = 0.0;	
	}

    ret = clcg_solver_cuda(cudaAx, cudaProgress, host_m, b, N, nz, &self_para, nullptr, cubHandle, cusHandle, CLCG_BICG);
    lcg_error_str(ret);

	std::clog << "Averaged error (compared with ans_x): " << avg_error(host_m, ans_x, N) << std::endl;

	// Solve with BICG_SYM
	for (size_t i = 0; i < N; i++)
	{
		host_m[i].x = 0.0; host_m[i].y = 0.0;	
	}

    ret = clcg_solver_cuda(cudaAx, cudaProgress, host_m, b, N, nz, &self_para, nullptr, cubHandle, cusHandle, CLCG_BICG_SYM);
    lcg_error_str(ret);

	std::clog << "Averaged error (compared with ans_x): " << avg_error(host_m, ans_x, N) << std::endl;

	// Free Host memory
	delete[] A;
	delete[] rowIdxA;
	delete[] colIdxA;
	delete[] b;
	delete[] ans_x;
	delete[] host_m;

	// Free Device memory
	cudaFree(d_A);
	cudaFree(d_rowIdxA);
	cudaFree(d_rowPtrA);
	cudaFree(d_colIdxA);
	cudaFree(d_B);

	cusparseDestroyDnVec(dvec_tmp);
	cusparseDestroySpMat(smat_A);
	cudaFree(d_buf);

	// Free handles
	cublasDestroy(cubHandle);
	cusparseDestroy(cusHandle);

	return 0;
}