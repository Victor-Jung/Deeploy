#ifndef __GAP_H__
#define __GAP_H__

typedef   signed short v2s __attribute__((vector_size (4)));
typedef unsigned short v2u __attribute__((vector_size (4)));

typedef   signed char  v4s __attribute__((vector_size (4)));
typedef unsigned char  v4u __attribute__((vector_size (4)));

// #ifdef __gap9__
// 	typedef   float16    v2h  __attribute__((vector_size (4)));
// 	typedef   float16alt v2ah __attribute__((vector_size (4)));
// 	typedef v2s V2S;
// #else
// 	#ifdef __FLOAT_EMUL__
// 		typedef float float16;
// 		typedef float float16alt;
// 		typedef float v2h __attribute__((vector_size (8)));
// 		typedef float v2ah __attribute__((vector_size (8)));
// 		typedef int V2S __attribute__((vector_size (8)));
// 	#else
// 		typedef short int float16;
// 		typedef short int float16alt;
// 		typedef short int v2h __attribute__((vector_size (4)));
// 		typedef short int v2ah __attribute__((vector_size (4)));
// 		typedef v2s V2S;
// 	#endif
// #endif

// typedef float16    f16;
// typedef float16alt f16a;

// #ifndef __EMUL__

// #if defined(__pulp__) || defined(__GAP8__) || defined(__GAP9__)
// #include "pmsis.h"
// #endif

// #if defined( __PULP_OS__)
// #include "rt/rt_api.h"
// #endif
typedef unsigned int rt_pointerT;
#else
typedef void * rt_pointerT;
#endif

#ifdef __AUTOTILER_APP__
// #include "at_api.h"
#endif

#include "GapBuiltins.h"
// #include "GapSystem.h"
// #endif