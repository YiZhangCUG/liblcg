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

#include "cmath"
#include "ctime"
#include "iostream"

#include "clcg_cudaf.h"


typedef int (*cuda_solver_ptr)(clcg_axfunc_cudaf_ptr Afp, clcg_progress_cudaf_ptr Pfp, cuComplex* m, 
    const cuComplex* B, const int n_size, const int nz_size, const clcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

int clbicg(clcg_axfunc_cudaf_ptr Afp, clcg_progress_cudaf_ptr Pfp, cuComplex* m, 
    const cuComplex* B, const int n_size, const int nz_size, const clcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

int clbicg_symmetric(clcg_axfunc_cudaf_ptr Afp, clcg_progress_cudaf_ptr Pfp, cuComplex* m, 
    const cuComplex* B, const int n_size, const int nz_size, const clcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

int clcg_solver_cuda(clcg_axfunc_cudaf_ptr Afp, clcg_progress_cudaf_ptr Pfp, cuComplex* m, const cuComplex* B, 
    const int n_size, const int nz_size, const clcg_para* param, void* instance, cublasHandle_t cub_handle, 
    cusparseHandle_t cus_handle, clcg_solver_enum solver_id)
{
    cuda_solver_ptr cg_solver;
    switch (solver_id)
	{
		case CLCG_BICG:
			cg_solver = clbicg;
			break;
		case CLCG_BICG_SYM:
			cg_solver = clbicg_symmetric;
			break;
		default:
			return CLCG_UNKNOWN_SOLVER;
	}

	return cg_solver(Afp, Pfp, m, B, n_size, nz_size, param, instance, cub_handle, cus_handle);
}

typedef int (*cuda_precondtioned_solver_ptr)(clcg_axfunc_cudaf_ptr Afp, clcg_axfunc_cudaf_ptr Mfp, clcg_progress_cudaf_ptr Pfp, 
    cuComplex* m, const cuComplex* B, const int n_size, const int nz_size, const clcg_para* param, 
    void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

int clpcg(clcg_axfunc_cudaf_ptr Afp, clcg_axfunc_cudaf_ptr Mfp, clcg_progress_cudaf_ptr Pfp, cuComplex* m, 
    const cuComplex* B, const int n_size, const int nz_size, const clcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

int clcg_solver_preconditioned_cuda(clcg_axfunc_cudaf_ptr Afp, clcg_axfunc_cudaf_ptr Mfp, clcg_progress_cudaf_ptr Pfp, 
    cuComplex* m, const cuComplex* B, const int n_size, const int nz_size, const clcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle, clcg_solver_enum solver_id)
{
    cuda_precondtioned_solver_ptr cgp_solver;
    switch (solver_id)
	{
		case CLCG_PCG:
			cgp_solver = clpcg; break;
		default:
			return CLCG_UNKNOWN_SOLVER;
	}

	return cgp_solver(Afp, Mfp, Pfp, m, B, n_size, nz_size, param, instance, cub_handle, cus_handle);
}

int clbicg(clcg_axfunc_cudaf_ptr Afp, clcg_progress_cudaf_ptr Pfp, cuComplex* m, 
    const cuComplex* B, const int n_size, const int nz_size, const clcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle)
{
    // set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;
    if (cub_handle == nullptr) return LCG_INVALID_POINTER;
    if (cus_handle == nullptr) return LCG_INVALID_POINTER;

	cuComplex *d_m = nullptr, *d_B = nullptr;
	cuComplex *r1k = nullptr, *r2k = nullptr;
	cuComplex *d1k = nullptr, *d2k = nullptr, *Ax = nullptr;
	cudaMalloc(&d_m, n_size * sizeof(cuComplex));
	cudaMalloc(&d_B, n_size * sizeof(cuComplex));
    cudaMalloc(&r1k, n_size * sizeof(cuComplex));
	cudaMalloc(&r2k, n_size * sizeof(cuComplex));
    cudaMalloc(&d1k, n_size * sizeof(cuComplex));
	cudaMalloc(&d2k, n_size * sizeof(cuComplex));
    cudaMalloc(&Ax, n_size * sizeof(cuComplex));

	// Copy initial solutions
	cudaMemcpy(d_m, m, n_size * sizeof(cuComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, n_size * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cusparseDnVecDescr_t dvec_m, dvec_d1k, dvec_d2k, dvec_Ax;
	cusparseCreateDnVec(&dvec_m, n_size, d_m, CUDA_C_32F);
	cusparseCreateDnVec(&dvec_d1k, n_size, d1k, CUDA_C_32F);
	cusparseCreateDnVec(&dvec_d2k, n_size, d2k, CUDA_C_32F);
	cusparseCreateDnVec(&dvec_Ax, n_size, Ax, CUDA_C_32F);

    cuComplex one, none;
    one.x = 1.0; one.y = 0.0;
    none.x = -1.0; none.y = 0.0;
	cuComplex ak, nak, conj_ak, Ad1d2, r1r2_next, betak, conj_betak;

	Afp(instance, cub_handle, cus_handle, dvec_m, dvec_Ax, n_size, nz_size, CUSPARSE_OPERATION_NON_TRANSPOSE);

    // r0 = B - Ax
    cudaMemcpy(r1k, d_B, n_size * sizeof(cuComplex), cudaMemcpyDeviceToDevice); // r0 = B
    cublasCaxpy_v2(cub_handle, n_size, &none, Ax, 1, r1k, 1); // r0 -= Ax
    cudaMemcpy(d1k, r1k, n_size * sizeof(cuComplex), cudaMemcpyDeviceToDevice); // d0 = r0

	clcg_vecC_conjugate(r1k, r2k, n_size);
	cudaMemcpy(d2k, r2k, n_size * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

	cuComplex r1r2;
    cublasCdotc_v2(cub_handle, n_size, r2k, 1, r1k, 1, &r1r2);

	float m_mod;
    if (!para.abs_diff)
    {
        cublasScnrm2_v2(cub_handle, n_size, d_m, 1, &m_mod);
        if (m_mod < 1.0) m_mod = 1.0;
    }

	float rk_mod;
	cublasScnrm2_v2(cub_handle, n_size, r1k, 1, &rk_mod);

	int ret, t = 0;
	if (para.abs_diff && rk_mod/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, rk_mod/n_size, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}	
	else if (rk_mod*rk_mod/(m_mod*m_mod) <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, rk_mod*rk_mod/(m_mod*m_mod), &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}

	float residual;
	while(1)
	{
		if (para.abs_diff) residual = rk_mod/n_size;
		else residual = rk_mod*rk_mod/(m_mod*m_mod);

		if (Pfp != nullptr)
		{
			if (Pfp(instance, d_m, residual, &para, n_size, nz_size, t))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

        Afp(instance, cub_handle, cus_handle, dvec_d1k, dvec_Ax, n_size, nz_size, CUSPARSE_OPERATION_NON_TRANSPOSE);
        
        cublasCdotc_v2(cub_handle, n_size, d2k, 1, Ax, 1, &Ad1d2);
        ak = cuCdivf(r1r2, Ad1d2);
        nak = cuCmulf(none, ak);
		conj_ak = cuConjf(nak);

        cublasCaxpy_v2(cub_handle, n_size, &ak, d1k, 1, d_m, 1);
        cublasCaxpy_v2(cub_handle, n_size, &nak, Ax, 1, r1k, 1);

        if (!para.abs_diff)
        {
            cublasScnrm2_v2(cub_handle, n_size, d_m, 1, &m_mod);
            if (m_mod < 1.0) m_mod = 1.0;
        }

        cublasScnrm2_v2(cub_handle, n_size, r1k, 1, &rk_mod);

		Afp(instance, cub_handle, cus_handle, dvec_d2k, dvec_Ax, n_size, nz_size, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);

		cublasCaxpy_v2(cub_handle, n_size, &conj_ak, Ax, 1, r2k, 1);

		cublasCdotc_v2(cub_handle, n_size, r2k, 1, r1k, 1, &r1r2_next);
		betak = cuCdivf(r1r2_next, r1r2);
		conj_betak = cuConjf(betak);
		r1r2 = r1r2_next;

        cublasCscal_v2(cub_handle, n_size, &betak, d1k, 1);
        cublasCaxpy_v2(cub_handle, n_size, &one, r1k, 1, d1k, 1);

		cublasCscal_v2(cub_handle, n_size, &conj_betak, d2k, 1);
        cublasCaxpy_v2(cub_handle, n_size, &one, r2k, 1, d2k, 1);
	}

	func_ends:
	{
		// Copy to host memories
		cudaMemcpy(m, d_m, n_size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

		cudaFree(d_m);
		cudaFree(d_B);
		cudaFree(r1k);
		cudaFree(r2k);
		cudaFree(d1k);
		cudaFree(d2k);	
		cudaFree(Ax);
        cusparseDestroyDnVec(dvec_m);
        cusparseDestroyDnVec(dvec_d1k);
		cusparseDestroyDnVec(dvec_d2k);
        cusparseDestroyDnVec(dvec_Ax);
	}

	return ret;
}

int clbicg_symmetric(clcg_axfunc_cudaf_ptr Afp, clcg_progress_cudaf_ptr Pfp, cuComplex* m, 
    const cuComplex* B, const int n_size, const int nz_size, const clcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle)
{
    // set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;
    if (cub_handle == nullptr) return LCG_INVALID_POINTER;
    if (cus_handle == nullptr) return LCG_INVALID_POINTER;

	cuComplex *d_m = nullptr, *d_B = nullptr;
	cuComplex *rk = nullptr, *dk = nullptr, *Ax = nullptr;
	cudaMalloc(&d_m, n_size * sizeof(cuDoubleComplex));
	cudaMalloc(&d_B, n_size * sizeof(cuDoubleComplex));
    cudaMalloc(&rk, n_size * sizeof(cuComplex));
    cudaMalloc(&dk, n_size * sizeof(cuComplex));
    cudaMalloc(&Ax, n_size * sizeof(cuComplex));

	// Copy initial solutions
	cudaMemcpy(d_m, m, n_size * sizeof(cuComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, n_size * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cusparseDnVecDescr_t dvec_m, dvec_dk, dvec_Ax;
	cusparseCreateDnVec(&dvec_m, n_size, d_m, CUDA_C_32F);
	cusparseCreateDnVec(&dvec_dk, n_size, dk, CUDA_C_32F);
	cusparseCreateDnVec(&dvec_Ax, n_size, Ax, CUDA_C_32F);

    cuComplex one, none;
    one.x = 1.0; one.y = 0.0;
    none.x = -1.0; none.y = 0.0;
	cuComplex ak, nak, rkrk2, betak, dkAx;

	Afp(instance, cub_handle, cus_handle, dvec_m, dvec_Ax, n_size, nz_size, CUSPARSE_OPERATION_NON_TRANSPOSE);

    // r0 = B - Ax
    cudaMemcpy(rk, d_B, n_size * sizeof(cuComplex), cudaMemcpyDeviceToDevice); // r0 = B
    cublasCaxpy_v2(cub_handle, n_size, &none, Ax, 1, rk, 1); // r0 -= Ax
    cudaMemcpy(dk, rk, n_size * sizeof(cuComplex), cudaMemcpyDeviceToDevice); // d0 = r0

	cuComplex rkrk;
    cublasCdotu_v2(cub_handle, n_size, rk, 1, rk, 1, &rkrk);

	float m_mod;
    if (!para.abs_diff)
    {
        cublasScnrm2_v2(cub_handle, n_size, d_m, 1, &m_mod);
        if (m_mod < 1.0) m_mod = 1.0;
    }

	float rk_mod;
	cublasScnrm2_v2(cub_handle, n_size, rk, 1, &rk_mod);

	int ret, t = 0;
	if (para.abs_diff && rk_mod/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, rk_mod/n_size, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}	
	else if (rk_mod*rk_mod/(m_mod*m_mod) <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, rk_mod*rk_mod/(m_mod*m_mod), &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}

	float residual;
	while(1)
	{
		if (para.abs_diff) residual = rk_mod/n_size;
		else residual = rk_mod*rk_mod/(m_mod*m_mod);

		if (Pfp != nullptr)
		{
			if (Pfp(instance, d_m, residual, &para, n_size, nz_size, t))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

        Afp(instance, cub_handle, cus_handle, dvec_dk, dvec_Ax, n_size, nz_size, CUSPARSE_OPERATION_NON_TRANSPOSE);
        
        cublasCdotu_v2(cub_handle, n_size, dk, 1, Ax, 1, &dkAx);
        ak = cuCdivf(rkrk, dkAx);
        nak = cuCmulf(none, ak);

        cublasCaxpy_v2(cub_handle, n_size, &ak, dk, 1, d_m, 1);
        cublasCaxpy_v2(cub_handle, n_size, &nak, Ax, 1, rk, 1);

        if (!para.abs_diff)
        {
            cublasScnrm2_v2(cub_handle, n_size, d_m, 1, &m_mod);
            if (m_mod < 1.0) m_mod = 1.0;
        }

        cublasScnrm2_v2(cub_handle, n_size, rk, 1, &rk_mod);

		cublasCdotu_v2(cub_handle, n_size, rk, 1, rk, 1, &rkrk2);
		betak = cuCdivf(rkrk2, rkrk);
		rkrk = rkrk2;

        cublasCscal_v2(cub_handle, n_size, &betak, dk, 1);
        cublasCaxpy_v2(cub_handle, n_size, &one, rk, 1, dk, 1);
	}

	func_ends:
	{
		// Copy to host memories
		cudaMemcpy(m, d_m, n_size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

		cudaFree(d_m);
		cudaFree(d_B);
		cudaFree(rk);
		cudaFree(dk);
		cudaFree(Ax);
        cusparseDestroyDnVec(dvec_m);
        cusparseDestroyDnVec(dvec_dk);
        cusparseDestroyDnVec(dvec_Ax);
	}

	return ret;
}

int clpcg(clcg_axfunc_cudaf_ptr Afp, clcg_axfunc_cudaf_ptr Mfp, clcg_progress_cudaf_ptr Pfp, cuComplex* m, 
    const cuComplex* B, const int n_size, const int nz_size, const clcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle)
{
    // set CGS parameters
	clcg_para para = (param != nullptr) ? (*param) : defparam2;

	//check parameters
	if (n_size <= 0) return CLCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return CLCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return CLCG_INVILAD_EPSILON;

	if (m == nullptr) return CLCG_INVALID_POINTER;
	if (B == nullptr) return CLCG_INVALID_POINTER;
    if (cub_handle == nullptr) return LCG_INVALID_POINTER;
    if (cus_handle == nullptr) return LCG_INVALID_POINTER;

	cuComplex *d_m = nullptr, *d_B = nullptr;
    cuComplex *rk = nullptr, *dk = nullptr, *sk = nullptr, *Ax = nullptr;
	cudaMalloc(&d_m, n_size * sizeof(cuComplex));
    cudaMalloc(&d_B, n_size * sizeof(cuComplex));
    cudaMalloc(&rk, n_size * sizeof(cuComplex));
    cudaMalloc(&dk, n_size * sizeof(cuComplex));
    cudaMalloc(&sk, n_size * sizeof(cuComplex));
    cudaMalloc(&Ax, n_size * sizeof(cuComplex));

	// Copy initial solutions
	cudaMemcpy(d_m, m, n_size * sizeof(cuComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, n_size * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cusparseDnVecDescr_t dvec_m, dvec_rk, dvec_dk, dvec_sk, dvec_Ax;
	cusparseCreateDnVec(&dvec_m, n_size, d_m, CUDA_C_32F);
    cusparseCreateDnVec(&dvec_rk, n_size, rk, CUDA_C_32F);
	cusparseCreateDnVec(&dvec_dk, n_size, dk, CUDA_C_32F);
    cusparseCreateDnVec(&dvec_sk, n_size, sk, CUDA_C_32F);
	cusparseCreateDnVec(&dvec_Ax, n_size, Ax, CUDA_C_32F);

    cuComplex one, none;
    one.x = 1.0; one.y = 0.0;
    none.x = -1.0; none.y = 0.0;
    cuComplex ak, nak, d_old, betak, dkAx;

    Afp(instance, cub_handle, cus_handle, dvec_m, dvec_Ax, n_size, nz_size, CUSPARSE_OPERATION_NON_TRANSPOSE);

    // r0 = B - Ax
    cudaMemcpy(rk, d_B, n_size * sizeof(cuComplex), cudaMemcpyDeviceToDevice); // r0 = B
    cublasCaxpy_v2(cub_handle, n_size, &none, Ax, 1, rk, 1); // r0 -= Ax

	Mfp(instance, cub_handle, cus_handle, dvec_rk, dvec_dk, n_size, nz_size, CUSPARSE_OPERATION_NON_TRANSPOSE);

	cuComplex d_new;
    cublasCdotu_v2(cub_handle, n_size, rk, 1, dk, 1, &d_new);

    float m_mod;
    if (!para.abs_diff)
    {
        cublasScnrm2_v2(cub_handle, n_size, d_m, 1, &m_mod);
        if (m_mod < 1.0) m_mod = 1.0;
    }

	float rk_mod;
	cublasScnrm2_v2(cub_handle, n_size, rk, 1, &rk_mod);

    int ret, t = 0;
	if (para.abs_diff && rk_mod/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, rk_mod/n_size, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}	
	else if (rk_mod*rk_mod/(m_mod*m_mod) <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, rk_mod*rk_mod/(m_mod*m_mod), &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}

	float residual;
	while(1)
	{
		if (para.abs_diff) residual = rk_mod/n_size;
		else residual = rk_mod*rk_mod/(m_mod*m_mod);

		if (Pfp != nullptr)
		{
			if (Pfp(instance, d_m, residual, &para, n_size, nz_size, t))
			{
				ret = CLCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = CLCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

        Afp(instance, cub_handle, cus_handle, dvec_dk, dvec_Ax, n_size, nz_size, CUSPARSE_OPERATION_NON_TRANSPOSE);
        cublasCdotu_v2(cub_handle, n_size, dk, 1, Ax, 1, &dkAx);
		ak = cuCdivf(d_new, dkAx);
        nak = cuCmulf(none, ak);

        cublasCaxpy_v2(cub_handle, n_size, &ak, dk, 1, d_m, 1);
        cublasCaxpy_v2(cub_handle, n_size, &nak, Ax, 1, rk, 1);

		if (!para.abs_diff)
        {
            cublasScnrm2_v2(cub_handle, n_size, d_m, 1, &m_mod);
            if (m_mod < 1.0) m_mod = 1.0;
        }

        cublasScnrm2_v2(cub_handle, n_size, rk, 1, &rk_mod);

        Mfp(instance, cub_handle, cus_handle, dvec_rk, dvec_sk, n_size, nz_size, CUSPARSE_OPERATION_NON_TRANSPOSE);

		d_old = d_new;
        cublasCdotu_v2(cub_handle, n_size, rk, 1, sk, 1, &d_new);

		betak = cuCdivf(d_new, d_old);

        cublasCscal_v2(cub_handle, n_size, &betak, dk, 1);
        cublasCaxpy_v2(cub_handle, n_size, &one, sk, 1, dk, 1);
	}

	func_ends:
	{
		// Copy to host memories
		cudaMemcpy(m, d_m, n_size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

		cudaFree(d_m);
		cudaFree(d_B);
		cudaFree(rk);
		cudaFree(dk);
		cudaFree(sk);
		cudaFree(Ax);
        cusparseDestroyDnVec(dvec_m);
        cusparseDestroyDnVec(dvec_rk);
        cusparseDestroyDnVec(dvec_dk);
        cusparseDestroyDnVec(dvec_sk);
        cusparseDestroyDnVec(dvec_Ax);
	}

	return ret;
}