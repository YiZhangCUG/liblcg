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
cuComplex one = {1.0, 0.0};
cuComplex zero = {0.0, 0.0};

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

float avg_error(cuComplex *a, cuComplex *b, int n)
{
	float avg = 0.0;
	cuComplex tmp;
	for (size_t i = 0; i < n; i++)
	{
		tmp = clcg_Cdiff(a[i], b[i]);
		avg += (tmp.x*tmp.x + tmp.y*tmp.y);
	}
	return sqrt(avg)/n;
}

class sample14 : public CLCG_CUDAF_Solver
{
public:
	sample14(){}
	virtual ~sample14(){}

	void solve(std::string inputPath, std::string answerPath, cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

	void AxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t)
	{
		// Calculate the product of A*x
		cusparseSpMV(cus_handle, oper_t, &one, smat_A, x, &zero, prod_Ax, CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);
		return;
	}

	void MxProduct(cublasHandle_t cub_handle, cusparseHandle_t cus_handle, cusparseDnVecDescr_t x, cusparseDnVecDescr_t prod_Ax, 
		const int n_size, const int nz_size, cusparseOperation_t oper_t)
	{
		cusparseSpSV_solve(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_L, x, dvec_p, 
			CUDA_C_32F, CUSPARSE_SPSV_ALG_DEFAULT, descr_L);
		
		cusparseSpSV_solve(cus_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, smat_L, dvec_p, prod_Ax, 
			CUDA_C_32F, CUSPARSE_SPSV_ALG_DEFAULT, descr_LT);
		return;
	}

private:
	int N, nz, lnz;
	int *rowIdxA, *colIdxA;
	cuDoubleComplex *A, *b;
	cuDoubleComplex *ans_x;
    cuComplex *Af, *bf;
	cuComplex *ans_xf;

	int *L_row, *L_col;
    cuComplex *L_val;

	void *d_buf, *d_buf2;
	cusparseSpMatDescr_t smat_A;
	cusparseSpMatDescr_t smat_L;
	cusparseSpSVDescr_t descr_L, descr_LT;

	int *d_rowIdxA; // COO
	int *d_rowPtrA; // CSR
	int *d_colIdxA;
	cuComplex *d_A;
	cuComplex *d_t;
	cuComplex *d_p;
	cusparseDnVecDescr_t dvec_p;

	int *d_rowIdxL; // COO
	int *d_rowPtrL; // CSR
	int *d_colIdxL;
	cuComplex *d_L;

	cuComplex *host_m;
	cusparseDnVecDescr_t dvec_tmp;
};

void sample14::solve(std::string inputPath, std::string answerPath, cublasHandle_t cub_handle, cusparseHandle_t cus_handle)
{
	read(inputPath, &N, &nz, &A, &rowIdxA, &colIdxA, &b);
	readAnswer(answerPath, &N, &ans_x);

    clcg_incomplete_Cholesky_cuda_half_buffsize(rowIdxA, colIdxA, nz, &lnz);

	std::clog << "N = " << N << std::endl;
	std::clog << "nz = " << nz << std::endl;
    std::clog << "lnz = " << lnz << std::endl;

    Af = new cuComplex [nz];
    bf = new cuComplex [nz];
    ans_xf = new cuComplex [nz];

	// Note that converting complex numbers from double to single precisions may case stack overflow
    for (size_t i = 0; i < nz; i++)
    {
        Af[i].x = A[i].x; Af[i].y = A[i].y;
        bf[i].x = b[i].x; bf[i].y = b[i].y;
        ans_xf[i].x = ans_x[i].x; ans_xf[i].y = ans_x[i].y;
    }
    
	L_row = new int [lnz];
    L_col = new int [lnz];
    L_val = new cuComplex [lnz];

    clcg_incomplete_Cholesky_cuda_half(rowIdxA, colIdxA, Af, N, nz, lnz, L_row, L_col, L_val);
/*
    for (size_t i = 0; i < lnz; i++)
    {
        std::cout << L_row[i] << " " << L_col[i] << " (" << L_val[i].x << "," << L_val[i].y << ")\n";
    }
*/
    // Allocate GPU memory & copy matrix/vector to device
	cudaMalloc(&d_A, nz * sizeof(cuComplex));
	cudaMalloc(&d_rowIdxA, nz * sizeof(int));
	cudaMalloc(&d_rowPtrA, (N + 1) * sizeof(int));
	cudaMalloc(&d_colIdxA, nz * sizeof(int));
	cudaMalloc(&d_t, N * sizeof(cuComplex));
	cudaMalloc(&d_p, N * sizeof(cuComplex));
    cusparseCreateDnVec(&dvec_p, N, d_p, CUDA_C_32F);

	cudaMemcpy(d_A, Af, nz * sizeof(cuComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowIdxA, rowIdxA, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIdxA, colIdxA, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_t, bf, N * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cudaMalloc(&d_L, lnz * sizeof(cuComplex));
	cudaMalloc(&d_rowIdxL, lnz * sizeof(int));
	cudaMalloc(&d_rowPtrL, (N + 1) * sizeof(int));
	cudaMalloc(&d_colIdxL, lnz * sizeof(int));

    cudaMemcpy(d_L, L_val, lnz * sizeof(cuComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowIdxL, L_row, lnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIdxL, L_col, lnz * sizeof(int), cudaMemcpyHostToDevice);

	// Convert matrix A from COO format to CSR format
	cusparseXcoo2csr(cus_handle, d_rowIdxA, nz, N, d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO);

	// Create sparse matrix
	cusparseCreateCsr(&smat_A, N, N, nz, d_rowPtrA, d_colIdxA, d_A, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);

	// Convert matrix L from COO format to CSR format
    cusparseXcoo2csr(cus_handle, d_rowIdxL, lnz, N, d_rowPtrL, CUSPARSE_INDEX_BASE_ZERO);

	// Create sparse matrix
    cusparseCreateCsr(&smat_L, N, N, lnz, d_rowPtrL, d_colIdxL, d_L, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);

    // Specify Lower fill mode.
    cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
	cusparseSpMatSetAttribute(smat_L, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));

	// Specify Non-Unit diagonal type.
    cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
	cusparseSpMatSetAttribute(smat_L, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));

    // This is just used to get bufferSize;
	cusparseCreateDnVec(&dvec_tmp, N, d_t, CUDA_C_32F);

	size_t bufferSize_B;
	cusparseSpMV_bufferSize(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_A,
		dvec_tmp, &zero, dvec_tmp, CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_B);

    // --- Start of the preconditioning part ---
    cusparseSpSV_createDescr(&descr_L);
    cusparseSpSV_createDescr(&descr_LT);

    size_t bufferSize, bufferSize_L, bufferSize_LT;
	bufferSize = bufferSize_B;

    cusparseSpSV_bufferSize(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_L, dvec_p, 
        dvec_tmp, CUDA_C_32F, CUSPARSE_SPSV_ALG_DEFAULT, descr_L, &bufferSize_L);
    cusparseSpSV_bufferSize(cus_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, smat_L, dvec_p, 
        dvec_tmp, CUDA_C_32F, CUSPARSE_SPSV_ALG_DEFAULT, descr_LT, &bufferSize_LT);

    bufferSize = max(max(bufferSize, bufferSize_L), bufferSize_LT);
	cudaMalloc(&d_buf, bufferSize);
	cudaMalloc(&d_buf2, bufferSize);

	cusparseSpSV_analysis(cus_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_L, dvec_tmp, dvec_p, 
		CUDA_C_32F, CUSPARSE_SPSV_ALG_DEFAULT, descr_L, d_buf);

	cusparseSpSV_analysis(cus_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, smat_L, dvec_p, dvec_tmp, 
		CUDA_C_32F, CUSPARSE_SPSV_ALG_DEFAULT, descr_LT, d_buf2);
	// --- End of the preconditioning part ---

	// Declare an initial solution
    clcg_para self_para = clcg_default_parameters();
	self_para.epsilon = 1e-6;
	self_para.abs_diff = 0;

	host_m = new cuComplex[N];

	// Preconditioning with incomplete-chelosky factorization
	for (size_t i = 0; i < N; i++)
	{
		host_m[i].x = 0.0; host_m[i].y = 0.0;	
	}

	MinimizePreconditioned(cub_handle, cus_handle, host_m, bf, N, nz, CLCG_PCG);

	std::clog << "Averaged error (compared with ans_x): " << avg_error(host_m, ans_xf, N) << std::endl;

	// Free Host memory
	if (rowIdxA != nullptr) delete[] rowIdxA;
	if (colIdxA != nullptr) delete[] colIdxA;
    if (A != nullptr) delete[] A;
	if (b != nullptr) delete[] b;
	if (ans_x != nullptr) delete[] ans_x;
    if (Af != nullptr) delete[] Af;
	if (bf != nullptr) delete[] bf;
	if (ans_xf != nullptr) delete[] ans_xf;

    if (L_row != nullptr) delete[] L_row;
    if (L_col != nullptr) delete[] L_col;
    if (L_val != nullptr) delete[] L_val;

    if (host_m != nullptr) delete[] host_m;

	cusparseDestroyDnVec(dvec_tmp);
    cusparseDestroyDnVec(dvec_p);

	cudaFree(d_buf);
	cudaFree(d_buf2);
	cudaFree(d_rowIdxA);
	cudaFree(d_rowPtrA);
	cudaFree(d_colIdxA);
    cudaFree(d_A);
	cudaFree(d_t);
	cudaFree(d_p);

    cudaFree(d_rowIdxL);
	cudaFree(d_rowPtrL);
	cudaFree(d_colIdxL);
    cudaFree(d_L);

	cusparseDestroySpMat(smat_A);
	cusparseDestroySpMat(smat_L);
    cusparseSpSV_destroyDescr(descr_L);
    cusparseSpSV_destroyDescr(descr_LT);
	return;
}

int main(int argc, char **argv)
{
	std::string inputPath = "data/case_1K_cA";
	std::string answerPath = "data/case_1K_cB";

	cublasHandle_t cubHandle;
	cusparseHandle_t cusHandle;

	cublasCreate(&cubHandle);
	cusparseCreate(&cusHandle);

	sample14 sp;
	sp.set_report_interval(100);
	sp.solve(inputPath, answerPath, cubHandle, cusHandle);

	cublasDestroy(cubHandle);
	cusparseDestroy(cusHandle);
	return 0;
}