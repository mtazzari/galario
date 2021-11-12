#include <sys/time.h>
#include <sys/resource.h>

struct timeval tuse;

#define CPU_TIME gettimeofday( &tuse, (struct timezone *)0 ); 

#define TCREATE(x)  \
    double __timerseconds##x=0; double  __timerstartseconds##x=0; \
    double __timerusec##x=0; double __timerstartusec##x=0; 

#define TCLEAR(x) {__timerseconds##x = 0; __timerusec##x = 0; }

#define TSTART(x) { CPU_TIME; \
        __timerstartseconds##x = tuse.tv_sec;  \
        __timerstartusec##x = tuse.tv_usec; }

#define TSTOP(x)  { CPU_TIME; \
        __timerseconds##x += (tuse.tv_sec - __timerstartseconds##x); \
        __timerusec##x    += (tuse.tv_usec - __timerstartusec##x);   }

#define TTIME(str,x) \
 printf("%s    %6.6f seconds \n", str, __timerseconds##x+__timerusec##x*1.0e-6);

#define TGIVE(x) (__timerseconds##x+__timerusec##x*1.0e-6)
