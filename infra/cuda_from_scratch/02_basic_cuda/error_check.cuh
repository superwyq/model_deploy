#pragma once
#include <stdio.h>
#define CHECK(result)                      \
do                                         \
{                                          \
	const cudaError_t error_code = result; \
	if (error_code != cudaSuccess)         \
	{                                      \
		printf("CUDA error:\n");           \
		printf("File:\t %s \n", __FILE__); \
		printf("Line:\t %d \n", __LINE__); \
		printf("Error code:\t %d\n", error_code);  \
		printf("Error text:\t %s\n", cudaGetErrorString(error_code));  \
		exit(1);  \
	}  \
} while (0);
//错误检测宏，接受运行时函数的返回，如果是错误则报错。