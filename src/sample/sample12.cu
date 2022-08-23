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
#include "../lib/preconditioner_cuda.h"

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

class sample12 : public CLCG_CUDA_Solver
{
public:
	sample12(){}
	virtual ~sample12(){}

	void solve(std::string inputPath, std::string answerPath, cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

	void AxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t)
	{
		// Calculate the product of A*x
		cusparseSpMV(cus_handle, oper_t, &one, smat_A, x, &zero, prod_Ax, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);
		return;
	}

	void MxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t)
	{
		cusparseSpSV_solve(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_IC, x, dvec_p, 
			CUDA_C_64F, CUSPARSE_SPSV_ALG_DEFAULT, descr_L);
		
		cusparseSpSV_solve(cus_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, smat_IC, dvec_p, prod_Ax, 
			CUDA_C_64F, CUSPARSE_SPSV_ALG_DEFAULT, descr_LT);
		return;
	}

private:
	int N, nz;
	int *rowIdxA, *colIdxA;
	cuDoubleComplex *A, *b;
	cuDoubleComplex *ans_x;

	int *IC_row, *IC_col;
    cuDoubleComplex *IC_val;

	void *d_buf, *d_buf2;
	cusparseSpMatDescr_t smat_A;
	cusparseSpMatDescr_t smat_IC;
	cusparseSpSVDescr_t descr_L, descr_LT;

	int *d_rowIdxA; // COO
	int *d_rowPtrA; // CSR
	int *d_colIdxA;
	cuDoubleComplex *d_A;
	cuDoubleComplex *d_p;
	cusparseDnVecDescr_t dvec_p;

	int *d_rowIdxIC; // COO
	int *d_rowPtrIC; // CSR
	int *d_colIdxIC;
	cuDoubleComplex *d_IC;

	cuDoubleComplex *host_m;
	cuDoubleComplex *d_t;
	cusparseDnVecDescr_t dvec_tmp;
};

void sample12::solve(std::string inputPath, std::string answerPath, cublasHandle_t cub_handle, cusparseHandle_t cus_handle)
{
	read(inputPath, &N, &nz, &A, &rowIdxA, &colIdxA, &b);
	readAnswer(answerPath, &N, &ans_x);

	std::clog << "N = " << N << std::endl;
	std::clog << "nz = " << nz << std::endl;

	IC_row = new int [nz];
    IC_col = new int [nz];
    IC_val = new cuDoubleComplex [nz];

    clcg_incomplete_Cholesky_cuda_full(rowIdxA, colIdxA, A, N, nz, IC_row, IC_col, IC_val);
/*
	for (size_t i = 0; i < nz; i++)
	{
		if (IC_row[i] >= IC_col[i])
		{
			std::cout << IC_row[i] << " " << IC_col[i] << " (" << IC_val[i].x << "," << IC_val[i].y << ")\n";	
		}
	}
*/
    // Allocate GPU memory & copy matrix/vector to device
	cudaMalloc(&d_A, nz * sizeof(cuDoubleComplex));
	cudaMalloc(&d_rowIdxA, nz * sizeof(int));
	cudaMalloc(&d_rowPtrA, (N + 1) * sizeof(int));
	cudaMalloc(&d_colIdxA, nz * sizeof(int));
	cudaMalloc(&d_p, N * sizeof(cuDoubleComplex));
    cusparseCreateDnVec(&dvec_p, N, d_p, CUDA_C_64F);

	cudaMemcpy(d_A, A, nz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowIdxA, rowIdxA, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIdxA, colIdxA, nz * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_IC, nz * sizeof(cuDoubleComplex));
	cudaMalloc(&d_rowIdxIC, nz * sizeof(int));
	cudaMalloc(&d_rowPtrIC, (N + 1) * sizeof(int));
	cudaMalloc(&d_colIdxIC, nz * sizeof(int));

    cudaMemcpy(d_IC, IC_val, nz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowIdxIC, IC_row, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIdxIC, IC_col, nz * sizeof(int), cudaMemcpyHostToDevice);

	// Convert matrix A from COO format to CSR format
	cusparseXcoo2csr(cus_handle, d_rowIdxA, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO);

	// Create sparse matrix
	cusparseCreateCsr(&smat_A, N, N, nz, d_rowPtrA, d_colIdxA, d_A, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

	// Convert matrix L from COO format to CSR format
    cusparseXcoo2csr(cus_handle, d_rowIdxIC, nz, N, d_rowPtrIC, CUSPARSE_INDEX_BASE_ZERO);

	// Create sparse matrix
    cusparseCreateCsr(&smat_IC, N, N, nz, d_rowPtrIC, d_colIdxIC, d_IC, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

	// Specify Non-Unit diagonal type.
    //cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
	//cusparseSpMatSetAttribute(smat_IC, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));

    // This is just used to get bufferSize;
	cudaMalloc(&d_t, N * sizeof(cuDoubleComplex));
	cusparseCreateDnVec(&dvec_tmp, N, d_t, CUDA_C_64F);

	size_t bufferSize_B;
	cusparseSpMV_bufferSize(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_A,
		dvec_tmp, &zero, dvec_tmp, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_B);

    // --- Start of the preconditioning part ---
    cusparseSpSV_createDescr(&descr_L);
    cusparseSpSV_createDescr(&descr_LT);

    size_t bufferSize, bufferSize_L, bufferSize_LT;
	bufferSize = bufferSize_B;

    cusparseSpSV_bufferSize(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_IC, dvec_p, 
        dvec_tmp, CUDA_C_64F, CUSPARSE_SPSV_ALG_DEFAULT, descr_L, &bufferSize_L);
    cusparseSpSV_bufferSize(cus_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, smat_IC, dvec_p, 
        dvec_tmp, CUDA_C_64F, CUSPARSE_SPSV_ALG_DEFAULT, descr_LT, &bufferSize_LT);

    bufferSize = max(max(bufferSize, bufferSize_L), bufferSize_LT);
	cudaMalloc(&d_buf, bufferSize);
	cudaMalloc(&d_buf2, bufferSize);

	cusparseSpSV_analysis(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_IC, dvec_tmp, dvec_p, 
		CUDA_C_64F, CUSPARSE_SPSV_ALG_DEFAULT, descr_L, d_buf);

	cusparseSpSV_analysis(cus_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, smat_IC, dvec_p, dvec_tmp, 
		CUDA_C_64F, CUSPARSE_SPSV_ALG_DEFAULT, descr_LT, d_buf2);

	// --- End of the preconditioning part ---

	// Declare an initial solution
    clcg_para self_para = clcg_default_parameters();
	self_para.epsilon = 1e-6;
	self_para.abs_diff = 0;

	host_m = new cuDoubleComplex[N];

	// Preconditioning with incomplete-chelosky factorization
	for (size_t i = 0; i < N; i++)
	{
		host_m[i].x = 0.0; host_m[i].y = 0.0;	
	}

	MinimizePreconditioned(cub_handle, cus_handle, host_m, b, N, nz, CLCG_PCG);

	std::clog << "Averaged error (compared with ans_x): " << avg_error(host_m, ans_x, N) << std::endl;

	// Free Host memory
	if (rowIdxA != nullptr) delete[] rowIdxA;
	if (colIdxA != nullptr) delete[] colIdxA;
    if (A != nullptr) delete[] A;
	if (b != nullptr) delete[] b;
	if (ans_x != nullptr) delete[] ans_x;

    if (IC_row != nullptr) delete[] IC_row;
    if (IC_col != nullptr) delete[] IC_col;
    if (IC_val != nullptr) delete[] IC_val;

    if (host_m != nullptr) delete[] host_m;

	cusparseDestroyDnVec(dvec_tmp);
    cusparseDestroyDnVec(dvec_p);

	cudaFree(d_buf);
	cudaFree(d_buf2);
	cudaFree(d_rowIdxA);
	cudaFree(d_rowPtrA);
	cudaFree(d_colIdxA);
    cudaFree(d_A);
	cudaFree(d_p);
	cudaFree(d_t);

    cudaFree(d_rowIdxIC);
	cudaFree(d_rowPtrIC);
	cudaFree(d_colIdxIC);
    cudaFree(d_IC);

	cusparseDestroySpMat(smat_A);
	cusparseDestroySpMat(smat_IC);
    cusparseSpSV_destroyDescr(descr_L);
    cusparseSpSV_destroyDescr(descr_LT);
	return;
}

int main(int argc, char **argv)
{
	std::string inputPath = "data/case_1M_cA";
	std::string answerPath = "data/case_1M_cB";

	cublasHandle_t cubHandle;
	cusparseHandle_t cusHandle;

	cublasCreate(&cubHandle);
	cusparseCreate(&cusHandle);

	sample12 sp;
	sp.set_report_interval(0);
	sp.solve(inputPath, answerPath, cubHandle, cusHandle);

	cublasDestroy(cubHandle);
	cusparseDestroy(cusHandle);
	return 0;
}