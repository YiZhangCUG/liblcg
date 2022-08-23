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

#include "lcg_cuda.h"


typedef int (*lcg_solver_cuda_ptr)(lcg_axfunc_cuda_ptr Afp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, 
    const int n_size, const int nz_size, const lcg_para* param, void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

int lcg(lcg_axfunc_cuda_ptr Afp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, const int n_size, const int nz_size, 
    const lcg_para* param, void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

int lcgs(lcg_axfunc_cuda_ptr Afp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, const int n_size, const int nz_size, 
    const lcg_para* param, void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle);


int lcg_solver_cuda(lcg_axfunc_cuda_ptr Afp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, const int n_size, const int nz_size, 
    const lcg_para* param, void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle, lcg_solver_enum solver_id)
{
	lcg_solver_cuda_ptr cg_solver_cuda;
	switch (solver_id)
	{
		case LCG_CG:
			cg_solver_cuda = lcg;
			break;
		case LCG_CGS:
			cg_solver_cuda = lcgs;
			break;
		default:
			cg_solver_cuda = lcg;
            break;
	}

	return cg_solver_cuda(Afp, Pfp, m, B, n_size, nz_size, param, instance, cub_handle, cus_handle);
}


int lpcg(lcg_axfunc_cuda_ptr Afp, lcg_axfunc_cuda_ptr Mfp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, 
	const int n_size, const int nz_size, const lcg_para* param, void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

int lcg_solver_preconditioned_cuda(lcg_axfunc_cuda_ptr Afp, lcg_axfunc_cuda_ptr Mfp, lcg_progress_cuda_ptr Pfp, 
    lcg_float* m, const lcg_float* B, const int n_size, const int nz_size, const lcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle, lcg_solver_enum solver_id)
{
	return lpcg(Afp, Mfp, Pfp, m, B, n_size, nz_size, param, instance, cub_handle, cus_handle);
}


int lpg(lcg_axfunc_cuda_ptr Afp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, 
	const lcg_float* low, const lcg_float* hig, const int n_size, const int nz_size, const lcg_para* param, 
	void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle);

int lcg_solver_constrained_cuda(lcg_axfunc_cuda_ptr Afp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, 
    const lcg_float* low, const lcg_float* hig, const int n_size, const int nz_size, const lcg_para* param, void* instance, 
    cublasHandle_t cub_handle, cusparseHandle_t cus_handle, lcg_solver_enum solver_id)
{
	return lpg(Afp, Pfp, m, B, low, hig, n_size, nz_size, param, instance, cub_handle, cus_handle);
}


int lcg(lcg_axfunc_cuda_ptr Afp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, const int n_size, 
    const int nz_size, const lcg_para* param, void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle)
{
	// set CG parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_EPSILON;

	if (m == nullptr) return LCG_INVALID_POINTER;
	if (B == nullptr) return LCG_INVALID_POINTER;
    if (cub_handle == nullptr) return LCG_INVALID_POINTER;
    if (cus_handle == nullptr) return LCG_INVALID_POINTER;

	// locate memory
	lcg_float *d_m = nullptr, *d_B = nullptr;
	lcg_float *gk = nullptr, *dk = nullptr, *Adk = nullptr;
	cudaMalloc(&d_m, n_size * sizeof(lcg_float));
    cudaMalloc(&d_B, n_size * sizeof(lcg_float));
	cudaMalloc(&gk, n_size * sizeof(lcg_float));
    cudaMalloc(&dk, n_size * sizeof(lcg_float));
    cudaMalloc(&Adk, n_size * sizeof(lcg_float));

	// Copy initial solutions
	cudaMemcpy(d_m, m, n_size * sizeof(lcg_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, n_size * sizeof(lcg_float), cudaMemcpyHostToDevice);

	cusparseDnVecDescr_t dvec_m, dvec_dk, dvec_Adk;
	cusparseCreateDnVec(&dvec_m, n_size, d_m, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_dk, n_size, dk, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_Adk, n_size, Adk, CUDA_R_64F);

    lcg_float none = -1.0;

	Afp(instance, cub_handle, cus_handle, dvec_m, dvec_Adk, n_size, nz_size);

    // g0 = Ax - B
    cudaMemcpy(gk, Adk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice); // g0 = A*x
    cublasDaxpy_v2(cub_handle, n_size, &none, d_B, 1, gk, 1); // g0 -= B
    cudaMemset(dk, 0, n_size * sizeof(lcg_float)); // d0 = 0
    cublasDaxpy_v2(cub_handle, n_size, &none, gk, 1, dk, 1); // d0 = -g0

	lcg_float m_mod;
	if (!para.abs_diff)
	{
		cublasDdot_v2(cub_handle, n_size, d_m, 1, d_m, 1, &m_mod); // m_mod = ||m||
		if (m_mod < 1.0) m_mod = 1.0;
	}

	lcg_float gk_mod;
    cublasDdot_v2(cub_handle, n_size, gk, 1, gk, 1, &gk_mod); // gk_mod = ||gk||

	int ret, t = 0;
	if (para.abs_diff && sqrt(gk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, sqrt(gk_mod)/n_size, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}
	else if (gk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, gk_mod/m_mod, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}

	lcg_float dTAd, ak, betak, gk1_mod, residual;
	while (1)
	{
		if (para.abs_diff) residual = sqrt(gk_mod)/n_size;
		else residual = gk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, d_m, residual, &para, n_size, nz_size, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

        Afp(instance, cub_handle, cus_handle, dvec_dk, dvec_Adk, n_size, nz_size);

        cublasDdot_v2(cub_handle, n_size, dk, 1, Adk, 1, &dTAd); // dTAd = dk^T * Adk
		ak = gk_mod/dTAd;

        cublasDaxpy_v2(cub_handle, n_size, &ak, dk, 1, d_m, 1); // m += ak*dk
        cublasDaxpy_v2(cub_handle, n_size, &ak, Adk, 1, gk, 1); // gk += ak*Adk

		if (!para.abs_diff)
		{
			cublasDdot_v2(cub_handle, n_size, d_m, 1, d_m, 1, &m_mod); // m_mod = ||m||
			if (m_mod < 1.0) m_mod = 1.0;
		}

        cublasDdot_v2(cub_handle, n_size, gk, 1, gk, 1, &gk1_mod); // gk1_mod = ||gk||
		betak = gk1_mod/gk_mod;
		gk_mod = gk1_mod;

        cublasDscal_v2(cub_handle, n_size, &betak, dk, 1); // dk *= betak
        cublasDaxpy_v2(cub_handle, n_size, &none, gk, 1, dk, 1); // dk -= gk
	}

	func_ends:
	{
		// Copy to host memories
		cudaMemcpy(m, d_m, n_size * sizeof(lcg_float), cudaMemcpyDeviceToHost);

		cudaFree(d_m);
		cudaFree(d_B);
        cudaFree(dk);
        cudaFree(gk);
        cudaFree(Adk);
		cusparseDestroyDnVec(dvec_m);
		cusparseDestroyDnVec(dvec_dk);
		cusparseDestroyDnVec(dvec_Adk);
	}

	return ret;
}

int lcgs(lcg_axfunc_cuda_ptr Afp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, const int n_size, 
    const int nz_size, const lcg_para* param, void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle)
{
	// set CG parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_EPSILON;

	if (m == nullptr) return LCG_INVALID_POINTER;
	if (B == nullptr) return LCG_INVALID_POINTER;
    if (cub_handle == nullptr) return LCG_INVALID_POINTER;
    if (cus_handle == nullptr) return LCG_INVALID_POINTER;

	// locate memory
	lcg_float *d_m = nullptr, *d_B = nullptr;
	lcg_float *rk = nullptr, *r0T = nullptr, *pk = nullptr, *qpk = nullptr;
	lcg_float *Ax = nullptr, *uk = nullptr,   *qk = nullptr, *wk = nullptr;
	cudaMalloc(&d_m, n_size * sizeof(lcg_float));
    cudaMalloc(&d_B, n_size * sizeof(lcg_float));
	cudaMalloc(&rk, n_size * sizeof(lcg_float));
    cudaMalloc(&r0T, n_size * sizeof(lcg_float));
    cudaMalloc(&pk, n_size * sizeof(lcg_float));
	cudaMalloc(&qpk, n_size * sizeof(lcg_float));
	cudaMalloc(&Ax, n_size * sizeof(lcg_float));
	cudaMalloc(&uk, n_size * sizeof(lcg_float));
	cudaMalloc(&qk, n_size * sizeof(lcg_float));
	cudaMalloc(&wk, n_size * sizeof(lcg_float));

	// Copy initial solutions
	cudaMemcpy(d_m, m, n_size * sizeof(lcg_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, n_size * sizeof(lcg_float), cudaMemcpyHostToDevice);

	cusparseDnVecDescr_t dvec_m, dvec_wk, dvec_pk, dvec_Ax;
	cusparseCreateDnVec(&dvec_m, n_size, d_m, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_wk, n_size, wk, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_pk, n_size, pk, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_Ax, n_size, Ax, CUDA_R_64F);

	lcg_float one = 1.0;
    lcg_float none = -1.0;

	Afp(instance, cub_handle, cus_handle, dvec_m, dvec_Ax, n_size, nz_size);

    // r0 = B - Ax
	cudaMemcpy(rk, d_B, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice); // r0 = B
	cublasDaxpy_v2(cub_handle, n_size, &none, Ax, 1, rk, 1); // r0 -= Ax
    // p0 = u0 = r0T = r0
	cudaMemcpy(pk, rk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(uk, rk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(r0T, rk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);

	lcg_float rkr0T;
	cublasDdot_v2(cub_handle, n_size, rk, 1, r0T, 1, &rkr0T);

	lcg_float m_mod;
    if (!para.abs_diff) 
	{
		cublasDdot_v2(cub_handle, n_size, d_m, 1, d_m, 1, &m_mod); // m_mod = ||m||
		if (m_mod < 1.0) m_mod = 1.0;
	}

	lcg_float rk_mod;
    cublasDdot_v2(cub_handle, n_size, rk, 1, rk, 1, &rk_mod); // rk_mod = ||rk||

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, sqrt(rk_mod)/n_size, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}
	else if (rk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, rk_mod/m_mod, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}

	lcg_float ak, nak, rkr0T1, AprT, betak, residual;
	while (1)
	{
		if (para.abs_diff) residual = sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, d_m, residual, &para, n_size, nz_size, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

        Afp(instance, cub_handle, cus_handle, dvec_pk, dvec_Ax, n_size, nz_size);

		AprT = 0.0;
		cublasDdot_v2(cub_handle, n_size, r0T, 1, Ax, 1, &AprT);
		ak = rkr0T/AprT;
		nak = -1.0*ak;

		cudaMemcpy(qk, uk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(wk, uk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
        cublasDaxpy_v2(cub_handle, n_size, &nak, Ax, 1, qk, 1);
        cublasDaxpy_v2(cub_handle, n_size, &one, qk, 1, wk, 1);

		Afp(instance, cub_handle, cus_handle, dvec_wk, dvec_Ax, n_size, nz_size);

		cublasDaxpy_v2(cub_handle, n_size, &ak, wk, 1, d_m, 1);
        cublasDaxpy_v2(cub_handle, n_size, &nak, Ax, 1, rk, 1);

		if (!para.abs_diff)
		{
			cublasDdot_v2(cub_handle, n_size, d_m, 1, d_m, 1, &m_mod);
			if (m_mod < 1.0) m_mod = 1.0;
		}

        cublasDdot_v2(cub_handle, n_size, rk, 1, rk, 1, &rk_mod);
		
		cublasDdot_v2(cub_handle, n_size, rk, 1, r0T, 1, &rkr0T1);
		betak = rkr0T1/rkr0T;
		rkr0T = rkr0T1;

		cudaMemcpy(uk, rk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
		cublasDaxpy_v2(cub_handle, n_size, &betak, qk, 1, uk, 1);

		cudaMemcpy(qpk, qk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
		cublasDaxpy_v2(cub_handle, n_size, &betak, pk, 1, qpk, 1);

		cudaMemcpy(pk, uk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
		cublasDaxpy_v2(cub_handle, n_size, &betak, qpk, 1, pk, 1);
	}

	func_ends:
	{
		// Copy to host memories
		cudaMemcpy(m, d_m, n_size * sizeof(lcg_float), cudaMemcpyDeviceToHost);

		cudaFree(d_m);
		cudaFree(d_B);
        cudaFree(rk);
		cudaFree(r0T);
		cudaFree(pk);
		cudaFree(qpk);
		cudaFree(Ax);
		cudaFree(uk);
		cudaFree(qk);
		cudaFree(wk);
		cusparseDestroyDnVec(dvec_m);
		cusparseDestroyDnVec(dvec_wk);
		cusparseDestroyDnVec(dvec_pk);
		cusparseDestroyDnVec(dvec_Ax);
	}

	return ret;
}

int lpcg(lcg_axfunc_cuda_ptr Afp, lcg_axfunc_cuda_ptr Mfp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, 
	const int n_size, const int nz_size, const lcg_para* param, void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle)
{
	// set CG parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	//check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_EPSILON;

	if (m == nullptr) return LCG_INVALID_POINTER;
	if (B == nullptr) return LCG_INVALID_POINTER;
    if (cub_handle == nullptr) return LCG_INVALID_POINTER;
    if (cus_handle == nullptr) return LCG_INVALID_POINTER;

	// locate memory
	lcg_float *d_m = nullptr, *d_B = nullptr;
	lcg_float *rk = nullptr, *zk = nullptr, *dk = nullptr, *Adk = nullptr;
	cudaMalloc(&d_m, n_size * sizeof(lcg_float));
    cudaMalloc(&d_B, n_size * sizeof(lcg_float));
	cudaMalloc(&rk, n_size * sizeof(lcg_float));
    cudaMalloc(&zk, n_size * sizeof(lcg_float));
    cudaMalloc(&dk, n_size * sizeof(lcg_float));
	cudaMalloc(&Adk, n_size * sizeof(lcg_float));

	// Copy initial solutions
	cudaMemcpy(d_m, m, n_size * sizeof(lcg_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, n_size * sizeof(lcg_float), cudaMemcpyHostToDevice);

	cusparseDnVecDescr_t dvec_m, dvec_rk, dvec_zk, dvec_dk, dvec_Adk;
	cusparseCreateDnVec(&dvec_m, n_size, d_m, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_rk, n_size, rk, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_zk, n_size, zk, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_dk, n_size, dk, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_Adk, n_size, Adk, CUDA_R_64F);

	lcg_float one = 1.0;
    lcg_float none = -1.0;

	Afp(instance, cub_handle, cus_handle, dvec_m, dvec_Adk, n_size, nz_size);

    // r0 = B - Ax
	cudaMemcpy(rk, d_B, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice); // r0 = B
	cublasDaxpy_v2(cub_handle, n_size, &none, Adk, 1, rk, 1); // r0 -= Ax

	Mfp(instance, cub_handle, cus_handle, dvec_rk, dvec_zk, n_size, nz_size);

    // d0 = z0
	cudaMemcpy(dk, zk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);

	lcg_float m_mod;
    if (!para.abs_diff) 
	{
		cublasDdot_v2(cub_handle, n_size, d_m, 1, d_m, 1, &m_mod); // m_mod = ||m||
		if (m_mod < 1.0) m_mod = 1.0;
	}

	lcg_float rk_mod;
    cublasDdot_v2(cub_handle, n_size, rk, 1, rk, 1, &rk_mod); // rk_mod = ||rk||

	lcg_float zTr;
	cublasDdot_v2(cub_handle, n_size, zk, 1, rk, 1, &zTr);

	int ret, t = 0;
	if (para.abs_diff && sqrt(rk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, sqrt(rk_mod)/n_size, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}
	else if (rk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, rk_mod/m_mod, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}

	lcg_float dTAd, ak, nak, betak, zTr1, residual;
	while (1)
	{
		if (para.abs_diff) residual = sqrt(rk_mod)/n_size;
		else residual = rk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, d_m, residual, &para, n_size, nz_size, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

        Afp(instance, cub_handle, cus_handle, dvec_dk, dvec_Adk, n_size, nz_size);

		cublasDdot_v2(cub_handle, n_size, dk, 1, Adk, 1, &dTAd);
		ak = zTr/dTAd;
		nak = -1.0*ak;

        cublasDaxpy_v2(cub_handle, n_size, &ak, dk, 1, d_m, 1);
        cublasDaxpy_v2(cub_handle, n_size, &nak, Adk, 1, rk, 1);

		Mfp(instance, cub_handle, cus_handle, dvec_rk, dvec_zk, n_size, nz_size);

		if (!para.abs_diff)
		{
			cublasDdot_v2(cub_handle, n_size, d_m, 1, d_m, 1, &m_mod);
			if (m_mod < 1.0) m_mod = 1.0;
		}

        cublasDdot_v2(cub_handle, n_size, rk, 1, rk, 1, &rk_mod);
		
		cublasDdot_v2(cub_handle, n_size, zk, 1, rk, 1, &zTr1);
		betak = zTr1/zTr;
		zTr = zTr1;

		cublasDscal_v2(cub_handle, n_size, &betak, dk, 1); // dk *= betak
		cublasDaxpy_v2(cub_handle, n_size, &one, zk, 1, dk, 1);
	}

	func_ends:
	{
		// Copy to host memories
		cudaMemcpy(m, d_m, n_size * sizeof(lcg_float), cudaMemcpyDeviceToHost);

		cudaFree(d_m);
		cudaFree(d_B);
        cudaFree(rk);
		cudaFree(zk);
		cudaFree(dk);
		cudaFree(Adk);
		cusparseDestroyDnVec(dvec_m);
		cusparseDestroyDnVec(dvec_rk);
		cusparseDestroyDnVec(dvec_zk);
		cusparseDestroyDnVec(dvec_dk);
		cusparseDestroyDnVec(dvec_Adk);
	}

	return ret;
}


int lpg(lcg_axfunc_cuda_ptr Afp, lcg_progress_cuda_ptr Pfp, lcg_float* m, const lcg_float* B, 
	const lcg_float* low, const lcg_float* hig, const int n_size, const int nz_size, const lcg_para* param, 
	void* instance, cublasHandle_t cub_handle, cusparseHandle_t cus_handle)
{
	// set CG parameters
	lcg_para para = (param != nullptr) ? (*param) : defparam;

	// check parameters
	if (n_size <= 0) return LCG_INVILAD_VARIABLE_SIZE;
	if (para.max_iterations < 0) return LCG_INVILAD_MAX_ITERATIONS;
	if (para.epsilon <= 0.0 || para.epsilon >= 1.0) return LCG_INVILAD_EPSILON;
	if (para.step <= 0.0) return LCG_INVALID_LAMBDA;

	if (m == nullptr) return LCG_INVALID_POINTER;
	if (B == nullptr) return LCG_INVALID_POINTER;
	if (low == nullptr) return LCG_INVALID_POINTER;
	if (hig == nullptr) return LCG_INVALID_POINTER;
	if (cub_handle == nullptr) return LCG_INVALID_POINTER;
    if (cus_handle == nullptr) return LCG_INVALID_POINTER;

	// locate memory
	lcg_float *d_m = nullptr, *d_B = nullptr;
	lcg_float *gk = nullptr, *Adk = nullptr;
	lcg_float *m_new = nullptr, *gk_new = nullptr;
	lcg_float *sk = nullptr, *yk = nullptr;
	cudaMalloc(&d_m, n_size * sizeof(lcg_float));
    cudaMalloc(&d_B, n_size * sizeof(lcg_float));
	cudaMalloc(&gk, n_size *sizeof(lcg_float));
	cudaMalloc(&Adk, n_size *sizeof(lcg_float));
	cudaMalloc(&m_new, n_size *sizeof(lcg_float));
	cudaMalloc(&gk_new, n_size *sizeof(lcg_float));
	cudaMalloc(&sk, n_size *sizeof(lcg_float));
	cudaMalloc(&yk, n_size *sizeof(lcg_float));

	// Copy initial solutions
	cudaMemcpy(d_m, m, n_size * sizeof(lcg_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, n_size * sizeof(lcg_float), cudaMemcpyHostToDevice);

	cusparseDnVecDescr_t dvec_m, dvec_mnew, dvec_Adk;
	cusparseCreateDnVec(&dvec_m, n_size, d_m, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_mnew, n_size, m_new, CUDA_R_64F);
	cusparseCreateDnVec(&dvec_Adk, n_size, Adk, CUDA_R_64F);

	lcg_float none = -1.0;
	lcg_float nalpha_k, alpha_k = para.step;

	lcg_set2box_cuda(low, hig, m, n_size);
	Afp(instance, cub_handle, cus_handle, dvec_m, dvec_Adk, n_size, nz_size);

	// g0 = Ax - B
    cudaMemcpy(gk, Adk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice); // g0 = A*x
    cublasDaxpy_v2(cub_handle, n_size, &none, d_B, 1, gk, 1); // g0 -= B

	lcg_float m_mod;
	if (!para.abs_diff)
	{
		cublasDdot_v2(cub_handle, n_size, d_m, 1, d_m, 1, &m_mod); // m_mod = ||m||
		if (m_mod < 1.0) m_mod = 1.0;
	}

	lcg_float gk_mod;
    cublasDdot_v2(cub_handle, n_size, gk, 1, gk, 1, &gk_mod); // gk_mod = ||gk||

	int ret, t = 0;
	if (para.abs_diff && sqrt(gk_mod)/n_size <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, sqrt(gk_mod)/n_size, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}
	else if (gk_mod/m_mod <= para.epsilon)
	{
		ret = LCG_ALREADY_OPTIMIZIED;
		if (Pfp != nullptr)
		{
			Pfp(instance, d_m, gk_mod/m_mod, &para, n_size, nz_size, 0);
		}
		goto func_ends;
	}


	lcg_float sk_mod, syk_mod, residual;
	while(1)
	{
		if (para.abs_diff) residual = sqrt(gk_mod)/n_size;
		else residual = gk_mod/m_mod;

		if (Pfp != nullptr)
		{
			if (Pfp(instance, d_m, residual, &para, n_size, nz_size, t))
			{
				ret = LCG_STOP; goto func_ends;
			}
		}

		if (residual <= para.epsilon)
		{
			ret = LCG_CONVERGENCE; goto func_ends;
		}

		if (para.max_iterations > 0 && t+1 > para.max_iterations)
		{
			ret = LCG_REACHED_MAX_ITERATIONS;
			break;
		}
		
		t++;

		nalpha_k = -1.0*alpha_k;
		cudaMemcpy(m_new, d_m, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
		cublasDaxpy_v2(cub_handle, n_size, &nalpha_k, gk, 1, m_new, 1);

		lcg_set2box_cuda(low, hig, m_new, n_size);
		Afp(instance, cub_handle, cus_handle, dvec_mnew, dvec_Adk, n_size, nz_size);
		
		cudaMemcpy(gk_new, Adk, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice); // g0 = A*x
    	cublasDaxpy_v2(cub_handle, n_size, &none, d_B, 1, gk, 1); // g0 -= B

		cudaMemcpy(sk, m_new, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
		cublasDaxpy_v2(cub_handle, n_size, &none, d_m, 1, sk, 1);

		cudaMemcpy(yk, gk_new, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
		cublasDaxpy_v2(cub_handle, n_size, &none, gk, 1, sk, 1);

		cublasDdot_v2(cub_handle, n_size, sk, 1, sk, 1, &sk_mod);
		cublasDdot_v2(cub_handle, n_size, sk, 1, yk, 1, &syk_mod);
		alpha_k = sk_mod/syk_mod;

		cudaMemcpy(d_m, m_new, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(gk, gk_new, n_size * sizeof(lcg_float), cudaMemcpyDeviceToDevice);

		if (!para.abs_diff)
		{
			cublasDdot_v2(cub_handle, n_size, d_m, 1, d_m, 1, &m_mod); // m_mod = ||m||
			if (m_mod < 1.0) m_mod = 1.0;
		}

		lcg_float gk_mod;
		cublasDdot_v2(cub_handle, n_size, gk, 1, gk, 1, &gk_mod); // gk_mod = ||gk||
	}

	func_ends:
	{
		// Copy to host memories
		cudaMemcpy(m, d_m, n_size * sizeof(lcg_float), cudaMemcpyDeviceToHost);

		cudaFree(d_m);
		cudaFree(d_B);
		cudaFree(gk);
		cudaFree(gk_new);
		cudaFree(m_new);
		cudaFree(sk);
		cudaFree(yk);
		cudaFree(Adk);
		cusparseDestroyDnVec(dvec_m);
		cusparseDestroyDnVec(dvec_mnew);
		cusparseDestroyDnVec(dvec_Adk);
	}

	return ret;
}