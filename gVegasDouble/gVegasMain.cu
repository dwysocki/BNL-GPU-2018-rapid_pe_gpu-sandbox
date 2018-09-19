#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

// includes, project
#include "helper_cuda.h"
// include initial files

#define __MAIN_LOGIC
#include "vegas.h"
#include "gvegas.h"
#undef __MAIN_LOGIC

#include "kernels.h"

double getrusage_sec()
{
   struct rusage t;
   struct timeval tv;
   getrusage(RUSAGE_SELF, &t);
   tv = t.ru_utime;
   return tv.tv_sec + (double)tv.tv_usec*1e-6;
}

int main(int argc, char** argv)
{

   //------------------
   //  Initialization
   //------------------
   //
   // program interface:
   //   program -n="ncall0" -i="itmx0" -a="nacc" -b="nBlockSize0"
   //
   // parameters:
   //   ncall = ncall0
   //   itmx  = itmx0
   //   acc   = nacc*0.00001f
   //   nBlockSize = nBlockSize0
   //

   int ncall0 = 0;
   int itmx0 = 10;
   int nacc  = 1;
   int nBlockSize0 = 256;
   int ndim0 = 6;
   int c;

   while ((c = getopt (argc, argv, "n:i:a:b:d:")) != -1)
       switch (c)
         {
         case 'n':
           ncall0 = atoi(optarg);
           break;
         case 'i':
           itmx0 = atoi(optarg);
           break;
         case 'a':
           nacc = atoi(optarg);
           break;
         case 'b':
           nBlockSize0 = atoi(optarg);
           break;
         case 'd':
           ndim0 = atoi(optarg);
           break;
         case '?':
           if (isprint (optopt))
             fprintf (stderr, "Unknown option `-%c'.\n", optopt);
           else
             fprintf (stderr,
                      "Unknown option character `\\x%x'.\n",
                      optopt);
           return 1;
         default:
           abort ();
         }

   //ncall = (1 << ncall0)*1024;
   ncall = ncall0; // more intuitive to users
   itmx = itmx0;
   acc = (double)nacc*0.000001;
   nBlockSize = nBlockSize0;
   ndim = ndim0;

   assert(ndim <= ndim_max);

   mds = 1;

   ng = 0;
   npg = 0;

   for (int i=0;i<ndim;i++) {
      xl[i] = 0.;
      xu[i] = 1.;
   }
   //If nprn = 1 it prints the whole work, when nprn = 0, just the text in this code
   //If nprn = -1, we can get the grid update information.

   nprn = 0;
//   nprn = -1;
//  nprn = 0;

   double avgi = 0.;
   double sd = 0.;
   double chi2a = 0.;

   gVegas(avgi, sd, chi2a);

   //-------------------------
   //  Print out information
   //-------------------------
   std::cout.clear();
   std::cout<<"#==========================="<<std::endl;
   std::cout<<"# No. of Thread Block Size : "<<nBlockSize<<std::endl;
   std::cout<<"#==========================="<<std::endl;
   std::cout<<"# No. of dimensions        : "<<ndim<<std::endl;
   std::cout<<"# No. of func calls / iter : "<<ncall<<std::endl;
   std::cout<<"# No. of max. iterations   : "<<itmx<<std::endl;
   std::cout<<"# Desired accuracy         : "<<acc<<std::endl;
   std::cout<<"#==========================="<<std::endl;
   std::cout<<"# Answer                   : "<<avgi<<" +- "<<sd<<std::endl;
   std::cout<<"# Chisquare                : "<<chi2a<<std::endl;
   std::cout<<"#==========================="<<std::endl;

   cudaThreadExit();

   //Print running times!
   std::cout<<"#==========================="<<std::endl;
   std::cout<<"# Function call time per iteration: " <<timeVegasCall/(double)it<<std::endl;
   std::cout<<"# Values moving time per iteration: " <<timeVegasMove/(double)it<<std::endl;
   std::cout<<"# Filling (reduce) time per iteration: " <<timeVegasFill/(double)it<<std::endl;
   std::cout<<"# Refining time per iteration: " <<timeVegasRefine/(double)it<<std::endl;
   std::cout<<"#==========================="<<std::endl;

   return 0;
}
