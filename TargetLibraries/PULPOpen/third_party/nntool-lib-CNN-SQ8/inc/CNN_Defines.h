#ifndef __CNN_DEFINED_H__
#define __CNN_DEFINED_H__

#ifdef __pulp__
#define Abs(a)          __builtin_pulp_abs((a))
#define Min(a, b)       __builtin_pulp_minsi((a), (b))
#define Max(a, b)       __builtin_pulp_maxsi((a), (b))
#else
#define Abs(a)          (((int)(a)<0)?(-(a)):(a))
#define Min(a, b)       (((a)<(b))?(a):(b))
#define Max(a, b)       (((a)>(b))?(a):(b))
#endif

#define Minu(a, b)            (( ((unsigned int)a)<((unsigned int)b) )?((unsigned int)a):((unsigned int)b) )

/* In the following n is the bound and x the value to be clamped */
/* R = Max(0, Min(x, n) */
#define AT_CLIP_POS(x, n)       gap_clipur((x), (n))

/* R = Max(0, Min(x, 2^(n-1)-1 */
#define AT_CLIP_POS_IMM(x, n)   gap_clipu((x), (n))

/* R = Max(n, Min(x, -(n+1)) */
#define AT_CLIP_(x, n)          gap_clipr((x), (n))

#define AT_CLIP(x, n)		gap_clip((x), (n))
#endif
