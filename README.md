# C++ Library of the Linear Conjugate Gradient Methods (LibLCG)

Yi Zhang (yizhang-geo@zju.edu.cn)

_School of Earth Sciences, Zhejiang University_

**This description only covers the brief introduction and use of the algorithm library. For more details, please refer to the code comments. If you still have questions, please email me. Interested students are also welcome to join the development team! **

## Introduction

liblcg is an efficient and extensible C++ linear conjugate gradient algorithm library. On the basis of the native data structure interface, it also provides algorithm interfaces based on Eigen3 and CUDA, which can easily realize accelerated computing based on CPU or GPU parallelism, among which The algorithm based on Eigen3 includes the implementation of dense and sparse matrix, while the algorithm based on CUDA is mainly the implementation of sparse matrix. liblcg includes several conjugate gradient algorithms for real and complex domains and several other iterative solvers. The existing methods include conjugate gradient method, pre-optimized conjugate gradient algorithm, conjugate gradient square algorithm, bistable conjugate gradient algorithm, BB step conjugate gradient projection method and SPG conjugate gradient projection method; Double conjugate gradient method, conjugate gradient flat method, pre-optimized conjugate gradient method and TFQMR method. The conjugate gradient method is widely used in unconstrained and inequality-constrained linear optimization problems, and has excellent convergence and computational efficiency.

The conjugate gradient algorithm can be used to solve systems of linear equations of the form:

````
Ax = B
````

Among them, A is a square matrix of order N, x is the model vector of size N\*1 to be solved, and B is the target vector of size N\*1 to be fitted. It should be noted that different kinds of conjugate gradient algorithms may have different requirements for A, such as being positive definite or symmetric. The specific requirements of different algorithms can be found in other references or in the comments in the code.

## Install

The algorithm library is assembled using the CMake tool, and the corresponding Makefile or project file can be generated on different operating platforms.

### compile options

The compilation options currently available for the algorithm library are:
* LibLCG_OPENMP: Whether to use OpenMP for acceleration, you need to install OpeMP. Default is ON.
* LibLCG_EIGEN: Whether to compile Eigen-based algorithms and excuses, Eigen needs to be installed. Default is ON.
* LibLCG_STD_COMPLEX: Whether to use std::complex\<double\> as the default type for complex numbers. Default is ON.
* LibLCG_CUDA: Whether to compile CUDA-based algorithms and excuses, CUDA needs to be installed. Default is ON.

Users can use the -D option in the cmake command to set compilation options, such as turning off LibLCG_Eigen:

```shell
cmake -DLibLCG_EIGEN=OFF
````

### Linux and MacOS

The default installation path for liblcg is /usr/local. Header files and dynamic libraries are installed in include and lib folders respectively. The specific compilation and installation steps are as follows:

1. Download and install CMake software;
2. Download and install the GCC compiler (common systems have built-in);
3. Use the following commands in the source file path to compile and install:

```shell
mkdir build && cd build && cmake .. && make install
````

### Windows

#### MinGW and GCC

The Windows system does not include the GNU compilation environment, and users need to download and configure it by themselves. Methods as below:

1. Download the MinGW installation file, and select gcc, pthreads and make related packages to install;
2. Download and install CMake software;
3. Add the CMake and MinGW executable paths to the Windows environment variables;
4. Use the following commands in the source file path to compile and install:

```shell
mkdir build && cd build && cmake .. -G "MinGW Makefiles" && make install
````

The default installation path is C:/Program\\ Files. Header files and dynamic libraries are installed in include and lib folders respectively.

**Note: Users need to manually add header files and dynamic library addresses to the computer's environment variables. **

#### Visual Studio

Users can use the CMake tool to build VS project files and compile and use dynamic libraries. Methods as below:

1. Download and install the Visual Studio software;
2. Download and install CMake software;
3. Use the following command in the source file path to generate the VS project file:

```shell
mkdir build && cd build && cmake .. -G "Visual Studio 16 2019"
````

_Note: If you need to generate other versions of VS project files, please use the -G command to view the corresponding identification code. _

4. Use Visual Studio to open the .sln project file and compile the dynamic library.

## use and compile

When users use library functions, they need to introduce corresponding header files into the source files, such as:

````cpp
#include "lcg/lcg.h"
````

The lcg dynamic library needs to be linked when compiling the executable. Take g++ as an example:

```shell
g++ example.cpp -llcg -o example_out
````

## quick start

To use liblcg to solve the linear equation system Ax=B, the user needs to define the calculation function (callback function) of the product of Ax, the function of which is to calculate the product Ax corresponding to different x. Taking the conjugate gradient algorithm of real number type as an example, the interface of its callback function is defined as:

````cpp
typedef void (*lcg_axfunc_ptr)(void* instance, const lcg_float* x, lcg_float* prod_Ax, const int n_size);
````

where `x` is the input vector, `prod_Ax` is the returned product vector, and `n` is the length of the two vectors. Note that matrix A is not included in the parameter list here, which means that A must be a global or class variable. The main reason for this design is that in the programming of some complex optimization problems, it is not practical or cost-effective to calculate and store A. At this time, the general strategy is to store the relevant variables and only calculate the product of Ax, so the matrix A is not always is existence.

After defining the Ax calculation function, the user can call the solver function lcg_solver() to solve the system of linear equations. Take the unconstrained solver function as an example, its declaration is as follows:

````cpp
int lcg_solver(lcg_axfunc_ptr Afp, lcg_progress_ptr Pfp, lcg_float* m, const lcg_float* B, const int n_size,
const lcg_para* param, void* instance, lcg_solver_enum solver_id = LCG_CGS);
````

in:
1. `lcg_axfunc_ptr Afp` is the callback function of forward calculation;
2. `lcg_progress_ptr Pfp` is the callback function for monitoring the iteration process (not required, just use the nullptr parameter when monitoring is not required);
3. `lcg_float* m` is the initial solution vector, and the solution obtained by iteration is also stored in this array;
4. `const lcg_float* B` Ax = item B in B;
5. `const int n_size` the size of the solution vector;
6. `const lcg_para* param` parameter used for iteration, this parameter is nullptr, that is, the default parameter is used;
7. The instance object passed in by `void* instance`, when this function is used in a class, it is the this pointer of the class, and when it is used in a normal function, it is nullptr;
8. `int solver_id` The solution method used by the solver function, the specific method code can be viewed in the corresponding header file;

### A simple example

````cpp
#include "cmath"
#include "iostream"
#include "lcg/lcg.h"

#define M 100
#define N 80

// Returns the maximum difference between two array elements
lcg_float max_diff(const lcg_float *a, const lcg_float *b, int size)
{
    lcg_float max = -1;
    for (int i = 0; i < size; i++)
    {
          max = lcg_max(sqrt((a[i] - b[i])*(a[i] - b[i])), max);
    }
    return max;
}

// Ordinary two-dimensional array as kernel matrix
lcg_float **kernel;
// intermediate result array
lcg_float *tmp_arr;

// Calculate the product of the kernel matrix multiplied by the vector lcg_solver's callback function
void CalAx(void* instance, const lcg_float* x, lcg_float* prod_Ax, const int n_s)
{
    // Note that the kernel matrix is ​​actually kernel^T * kernel, the size is N*N
    lcg_matvec(kernel, x, tmp_arr, M, n_s, MatNormal); // tmp_tar = kernel * x
    lcg_matvec(kernel, tmp_arr, prod_Ax, M, n_s, MatTranspose); // prod_Ax = kernel^T * tmp_tar
    return;
}

// Define the callback function of the monitoring function lcg_solver
// This function displays the current iteration count and convergence value
int Prog(void* instance, const lcg_float* m, const lcg_float converge, const lcg_para* param, const int n_s, const int k)
{
    std::clog << "\rIteration-times: " << k << "\tconvergence: " << converge;
    return 0;
}

int main(int argc, char const *argv[])
{
    // open up the array space
    kernel = lcg_malloc(M, N);
    tmp_arr = lcg_malloc(M);

    // assign initial value to kernel matrix
    lcg_vecrnd(kernel, -1.0, 1.0, M, N);

    // generate a set of theoretical solutions
    lcg_float *fm = lcg_malloc(N);
    lcg_vecrnd(fm, 1.0, 2.0, N);

    // Calculate the conjugate gradient B term
    lcg_float *B = lcg_malloc(N);
    lcg_matvec(kernel, fm, tmp_arr, M, N, MatNormal);
    lcg_matvec(kernel, tmp_arr, B, M, N, MatTranspose);

    // set conjugate gradient parameters
    lcg_para self_para = lcg_default_parameters();
    self_para.epsilon = 1e-5;
    self_para.abs_diff = 0;

    // declare a set of solutions
    lcg_float *m = lcg_malloc(N);
    lcg_vecset(m, 0.0, N);

    // Solve the system of linear equations using the standard conjugate gradient method (LCG_CG)
    // Pass the callback function to the solver
    // Since the callback function is a global function, the value of the instance variable is NULL
    int ret = lcg_solver(CalAx, Prog, m, B, N, &self_para, NULL, LCG_CG);
    std::clog << std::endl; lcg_error_str(ret);
    std::clog << "maximal difference: " << max_diff(fm, m, N) << std::endl << std::endl;

    // destroy the array
    lcg_free(kernel, M);
    lcg_free(tmp_arr);
    lcg_free(fm);
    lcg_free(B);
    lcg_free(m);
    return 0;
}
````

**Complete examples are stored in the [sample](src/sample) folder. **

## class template

liblcg defines a general solution class template for different types of conjugate gradient algorithms, including pointer proxy for functions in the class and general monitoring function implementation, which can be directly inherited and used by users. It should be noted that these class templates define pure virtual function interfaces, and users need to implement them all. The unused functions can be defined as empty functions. Taking the real number solution class template as an example, the interface functions that need to be implemented include:

````cpp
void AxProduct(const lcg_float* a, lcg_float* b, const int num) = 0
void MxProduct(const lcg_float* a, lcg_float* b, const int num) = 0
````

Where `AxProduct` is the calculation function of Ax, and `MxProduct` is the calculation function of the pre-optimization process, that is, M^-1x.
