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

#include "../lib/solver_cuda.h"

// Declare as global variables
cuDoubleComplex one = {1.0, 0.0};
cuDoubleComplex zero = {0.0, 0.0};

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

class sample10 : public CLCG_CUDA_Solver
{
public:
	sample10(){}
	virtual ~sample10(){}

	void solve(std::string inputPath, std::string answerPath);

	void AxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
    cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, const int n_size, const int nz_size, 
	cusparseOperation_t oper_t)
	{
		// Calculate the product of A*x
		cusparseSpMV(cus_handle, oper_t, &one, smat_A, x, &zero, prod_Ax, CUDA_C_64F, CUSPARSE_MV_ALG_DEFAULT, d_buf);
		return;
	}

	void MxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, 
		cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, const int n_size, const int nz_size, 
		cusparseOperation_t oper_t)
	{
		void *d_x, *d_Ax;
		cusparseDnVecGetValues(x, &d_x);
		cusparseDnVecGetValues(prod_Ax, &d_Ax);

		if (use_incomplete_cholesky)
		{
			cusparseZcsrsv2_solve(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_size, nz_size, &one, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_L, (cuDoubleComplex*) d_x, (cuDoubleComplex*) d_pd, 
				CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);

			cusparseZcsrsv2_solve(cus_handle, CUSPARSE_OPERATION_TRANSPOSE, n_size, nz_size, &one, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_LT, (cuDoubleComplex*) d_pd, (cuDoubleComplex*) d_Ax, 
				CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);
		}
		else
		{
			clcg_vecDvecZ_element_wise((cuDoubleComplex*) d_x, d_pd, (cuDoubleComplex*) d_Ax, n_size);
		}	
		return;
	}

private:
	bool use_incomplete_cholesky;

	int N, nz;
	int *rowIdxA, *colIdxA;
	cuDoubleComplex *A, *b;
	cuDoubleComplex *ans_x;

	void *d_buf;
	cusparseSpMatDescr_t smat_A;

	int *d_rowIdxA; // COO
	int *d_rowPtrA; // CSR
	int *d_colIdxA;
	cuDoubleComplex *d_A;
	cuDoubleComplex *d_pd;
	cuDoubleComplex *d_ic;

	cusparseMatDescr_t descr_A;
	cusparseMatDescr_t descr_L;
	csric02Info_t icinfo_A;
	csrsv2Info_t info_L;
	csrsv2Info_t info_LT;

	cuDoubleComplex *host_m;
	cusparseDnVecDescr_t dvec_tmp;
};

void sample10::solve(std::string inputPath, std::string answerPath)
{
	read(inputPath, &N, &nz, &A, &rowIdxA, &colIdxA, &b);
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
	cudaMalloc(&d_pd, N * sizeof(cuDoubleComplex));

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
	cusparseCreateDnVec(&dvec_tmp, N, d_pd, CUDA_C_64F);

	size_t bufferSize_B;
	cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_A,
		dvec_tmp, &zero, dvec_tmp, CUDA_C_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize_B);

	// --- Start of the preconditioning part ---
	// Get the diagonal elemenets
	clcg_smZcsr_get_diagonal(d_rowPtrA, d_colIdxA, d_A, N, d_pd);

	// Copy A
	cudaMalloc(&d_ic, nz * sizeof(cuDoubleComplex));
	cudaMemcpy(d_ic, d_A, nz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

	// create descriptor for matrix A
	cusparseCreateMatDescr(&descr_A);

	// initialize properties of matrix A
	cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
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

	int bufferSize, bufferSize_A, bufferSize_L, bufferSize_LT;
	bufferSize = bufferSize_B;

	// Compute buffer size in computing ic factorization
	cusparseZcsric02_bufferSize(cusHandle, N, nz, descr_A, d_A, d_rowPtrA, 
		d_colIdxA, icinfo_A, &bufferSize_A);
	cusparseZcsrsv2_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
		N, nz, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_L, &bufferSize_L);
	cusparseZcsrsv2_bufferSize(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, 
		N, nz, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_LT, &bufferSize_LT);
	
	bufferSize = max(max(max(bufferSize, bufferSize_A), bufferSize_L), bufferSize_LT);
	cudaMalloc(&d_buf, bufferSize);

	// Perform incomplete-choleskey factorization: analysis phase
	cusparseZcsric02_analysis(cusHandle, N, nz, descr_A, d_ic, d_rowPtrA, 
		d_colIdxA, icinfo_A, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);
	cusparseZcsrsv2_analysis(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
		N, nz, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_L, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);
	cusparseZcsrsv2_analysis(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, 
		N, nz, descr_L, d_ic, d_rowPtrA, d_colIdxA, info_LT, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);

	// Perform incomplete-choleskey factorization: solve phase
	cusparseZcsric02(cusHandle, N, nz, descr_A, d_ic, d_rowPtrA, d_colIdxA, 
		icinfo_A, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf);
	// --- End of the preconditioning part ---

	// Declare an initial solution
	host_m = new cuDoubleComplex[N];

    clcg_para self_para = clcg_default_parameters();
	self_para.epsilon = 1e-6;

	// Preconditioning with Diagonal elements
	for (size_t i = 0; i < N; i++)
	{
		host_m[i].x = 0.0; host_m[i].y = 0.0;	
	}

	use_incomplete_cholesky = false;
	MinimizePreconditioned(cubHandle, cusHandle, host_m, b, N, nz, CLCG_PCG);

	std::clog << "Averaged error (compared with ans_x): " << avg_error(host_m, ans_x, N) << std::endl;
	
	// Preconditioning with incomplete-Cholesky factorization
	for (size_t i = 0; i < N; i++)
	{
		host_m[i].x = 0.0; host_m[i].y = 0.0;	
	}

	use_incomplete_cholesky = true;
	MinimizePreconditioned(cubHandle, cusHandle, host_m, b, N, nz, CLCG_PCG);

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
	return;
}

int main(int argc, char **argv)
{
	std::string inputPath = "data/case_10K_cA";
	std::string answerPath = "data/case_10K_cB";

	sample10 sp;
	sp.set_report_interval(0);
	sp.solve(inputPath, answerPath);
	return 0;
}