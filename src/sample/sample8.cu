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

#include "../lib/lcg_cuda.h"

void read(std::string filePath, int *pN, int *pnz, double **cooVal,
	int **cooRowIdx, int **cooColIdx, double **b)
{
	std::ifstream in(filePath, std::ios::binary);

	in.read((char*)pN, sizeof(int));
	in.read((char*)pnz, sizeof(int));

	*cooVal = new double[*pnz]{};
	*cooRowIdx = new int[*pnz]{};
	*cooColIdx = new int[*pnz]{};
	*b = new double[*pN]{};

	for (int i = 0; i < *pnz; ++i)
	{
		in.read((char*)&(*cooRowIdx)[i], sizeof(int));
		in.read((char*)&(*cooColIdx)[i], sizeof(int));
		in.read((char*)&(*cooVal)[i], sizeof(double));
	}

	in.read((char*)(*b), sizeof(double)*(*pN));
    return;
}

void readAnswer(std::string filePath, int *pN, double **x)
{
	std::ifstream in(filePath, std::ios::binary);

	in.read((char*)pN, sizeof(int));

	*x = new double[*pN]{};

	in.read((char*)(*x), sizeof(double)*(*pN));
    return;
}

lcg_float avg_error(lcg_float *a, lcg_float *b, int n)
{
	lcg_float avg = 0.0;
	for (size_t i = 0; i < n; i++)
	{
		avg += (a[i] - b[i])*(a[i] - b[i]);
	}
	return sqrt(avg)/n;
}

// Declare as global variables
lcg_float one = 1.0;
lcg_float zero = 0.0;

void *d_buf;
cusparseSpMatDescr_t smat_A;

int *d_rowIdxA; // COO
int *d_rowPtrA; // CSR
int *d_colIdxA;
double *d_A;
double *d_pd;
double *d_ic;

cusparseMatDescr_t descr_A = 0;
cusparseMatDescr_t descr_L = 0;
csric02Info_t icinfo_A = 0;
csrsv2Info_t info_L = 0;
csrsv2Info_t info_LT = 0;

void cudaAx(void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
    cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, const int n_size, const int nz_size)
{
	// Calculate the product of A*x
	cusparseSpMV(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_A,
		x, &zero, prod_Ax, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, d_buf);
    return;
}

void cudaMx(void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
    cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, const int n_size, const int nz_size)
{
	void *d_x, *d_Ax;
	cusparseDnVecGetValues(x, &d_x);
	cusparseDnVecGetValues(prod_Ax, &d_Ax);

	cusparseDcsrsv2_solve(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
		n_size, nz_size, &one, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_L, (double*) d_x, (double*) d_pd, 
		CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);

	cusparseDcsrsv2_solve(cus_handle, CUSPARSE_OPERATION_TRANSPOSE, 
		n_size, nz_size, &one, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_LT, (double*) d_pd, (double*) d_Ax, 
		CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);
    return;
}

int cudaProgress(void* instance, const lcg_float* m, const lcg_float converge, 
	const lcg_para* param, const int n_size, const int nz_size, const int k)
{
    if (converge <= param->epsilon) {
		std::clog << "Iteration-times: " << k << "\tconvergence: " << converge << std::endl;
	}
	return 0;
}

int main(int argc, char **argv)
{
	std::string inputPath = "data/case_10K_A";
	std::string answerPath = "data/case_10K_B";

	int N;
	int nz;
	double *A;
	int *rowIdxA;
	int *colIdxA;
	double *b;
	read(inputPath, &N, &nz, &A, &rowIdxA, &colIdxA, &b);

	double *ans_x;
	readAnswer(answerPath, &N, &ans_x);

	std::clog << "N = " << N << std::endl;
	std::clog << "nz = " << nz << std::endl;
	
	// Create handles
	cublasHandle_t cubHandle;
	cusparseHandle_t cusHandle;

	cublasCreate(&cubHandle);
	cusparseCreate(&cusHandle);

	// Allocate GPU memory & copy matrix/vector to device
	cudaMalloc(&d_A, nz * sizeof(double));
	cudaMalloc(&d_rowIdxA, nz * sizeof(int));
	cudaMalloc(&d_rowPtrA, (N + 1) * sizeof(int));
	cudaMalloc(&d_colIdxA, nz * sizeof(int));
	cudaMalloc(&d_pd, N * sizeof(double));

	cudaMemcpy(d_A, A, nz * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowIdxA, rowIdxA, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIdxA, colIdxA, nz * sizeof(int), cudaMemcpyHostToDevice);

	// Convert matrix A from COO format to CSR format
	cusparseXcoo2csr(cusHandle, d_rowIdxA, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO);

	// Create sparse matrix
	cusparseCreateCsr(&smat_A, N, N, nz, d_rowPtrA, d_colIdxA, d_A, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

	// This is just used to get bufferSize;
	cusparseDnVecDescr_t dvec_tmp;
	cusparseCreateDnVec(&dvec_tmp, N, d_pd, CUDA_R_64F);

	size_t bufferSize_B;
	cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_A,
		dvec_tmp, &zero, dvec_tmp, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize_B);

	// --- Start of the preconditioning part ---

	// Copy A
	cudaMalloc(&d_ic, nz * sizeof(lcg_float));
	cudaMemcpy(d_ic, d_A, nz * sizeof(lcg_float), cudaMemcpyDeviceToDevice);

	int bufferSize, bufferSize_A, bufferSize_L, bufferSize_LT;
	bufferSize = bufferSize_B;

	// create descriptor for matrix A
	cusparseCreateMatDescr(&descr_A);

	// initialize properties of matrix A
	cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descr_A, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descr_A, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);

	// create descriptor for matrix L
	cusparseCreateMatDescr(&descr_L);

	// initialize properties of matrix L
	cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);

	// Create empty info objects for incomplete-cholesky factorization
	cusparseCreateCsric02Info(&icinfo_A);
	cusparseCreateCsrsv2Info(&info_L);
	cusparseCreateCsrsv2Info(&info_LT);

	// Compute buffer size in computing ic factorization
	cusparseDcsric02_bufferSize(cusHandle, N, nz, descr_A, d_A, d_rowPtrA, 
		d_colIdxA, icinfo_A, &bufferSize_A);
	cusparseDcsrsv2_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
		N, nz, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_L, &bufferSize_L);
	cusparseDcsrsv2_bufferSize(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, 
		N, nz, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_LT, &bufferSize_LT);
	
	bufferSize = max(max(max(bufferSize, bufferSize_A), bufferSize_L), bufferSize_LT);
	cudaMalloc(&d_buf, bufferSize);

	// Perform incomplete-choleskey factorization: analysis phase
	cusparseDcsric02_analysis(cusHandle, N, nz, descr_A, d_ic, d_rowPtrA, 
		d_colIdxA, icinfo_A, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);
	cusparseDcsrsv2_analysis(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
		N, nz, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_L, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);
	cusparseDcsrsv2_analysis(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, 
		N, nz, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_LT, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);

	// Perform incomplete-choleskey factorization: solve phase
	cusparseDcsric02(cusHandle, N, nz, descr_A, d_ic, d_rowPtrA, d_colIdxA, 
		icinfo_A, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);

	// --- End of the preconditioning part ---

	// Declare an initial solution
    lcg_para self_para = lcg_default_parameters();
	self_para.epsilon = 1e-6;
	self_para.abs_diff = 0;

	int ret;
	double *host_m = new double[N];

	// Solve with CG
	for (size_t i = 0; i < N; i++)
	{
		host_m[i] = 0.0;
	}

    ret = lcg_solver_cuda(cudaAx, cudaProgress, host_m, b, N, nz, &self_para, nullptr, cubHandle, cusHandle, LCG_CG);
    lcg_error_str(ret);

	std::clog << "Averaged error (compared with ans_x): " << avg_error(host_m, ans_x, N) << std::endl;

	// Solve with CGS
	for (size_t i = 0; i < N; i++)
	{
		host_m[i] = 0.0;
	}

	ret = lcg_solver_cuda(cudaAx, cudaProgress, host_m, b, N, nz, &self_para, nullptr, cubHandle, cusHandle, LCG_CGS);
    lcg_error_str(ret);

	std::clog << "Averaged error (compared with ans_x): " << avg_error(host_m, ans_x, N) << std::endl;

	// Solve with PCG
	for (size_t i = 0; i < N; i++)
	{
		host_m[i] = 0.0;
	}

	ret = lcg_solver_preconditioned_cuda(cudaAx, cudaMx, cudaProgress, host_m, b, N, nz, &self_para, nullptr, cubHandle, cusHandle, LCG_PCG);
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
	cudaFree(d_pd);
	cudaFree(d_ic);

	cusparseDestroyDnVec(dvec_tmp);
	cusparseDestroySpMat(smat_A);
	cudaFree(d_buf);

	cusparseDestroyMatDescr(descr_A);
	cusparseDestroyMatDescr(descr_L);
	cusparseDestroyCsric02Info(icinfo_A);
	cusparseDestroyCsrsv2Info(info_L);
	cusparseDestroyCsrsv2Info(info_LT);

	// Free handles
	cublasDestroy(cubHandle);
	cusparseDestroy(cusHandle);

	return 0;
}