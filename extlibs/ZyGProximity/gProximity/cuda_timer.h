/*
 *  gProximity Library.
 *  
 *  
 *  Copyright (C) 2010 University of North Carolina at Chapel Hill.
 *  All rights reserved.
 *  
 *  Permission to use, copy, modify, and distribute this software and its
 *  documentation for educational, research, and non-profit purposes, without
 *  fee, and without a written agreement is hereby granted, provided that the
 *  above copyright notice, this paragraph, and the following four paragraphs
 *  appear in all copies.
 *  
 *  Permission to incorporate this software into commercial products may be
 *  obtained by contacting the University of North Carolina at Chapel Hill.
 *  
 *  This software program and documentation are copyrighted by the University of
 *  North Carolina at Chapel Hill. The software program and documentation are
 *  supplied "as is", without any accompanying services from the University of
 *  North Carolina at Chapel Hill or the authors. The University of North
 *  Carolina at Chapel Hill and the authors do not warrant that the operation of
 *  the program will be uninterrupted or error-free. The end-user understands
 *  that the program was developed for research purposes and is advised not to
 *  rely exclusively on the program for any reason.
 *  
 *  IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL OR ITS
 *  EMPLOYEES OR THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 *  SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 *  ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE
 *  UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL OR THE AUTHORS HAVE BEEN ADVISED
 *  OF THE POSSIBILITY OF SUCH DAMAGE.
 *  
 *  THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND THE AUTHORS SPECIFICALLY
 *  DISCLAIM ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND ANY
 *  STATUTORY WARRANTY OF NON-INFRINGEMENT. THE SOFTWARE PROVIDED HEREUNDER IS
 *  ON AN "AS IS" BASIS, AND THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND
 *  THE AUTHORS HAVE NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
 *  ENHANCEMENTS, OR MODIFICATIONS.
 *  
 *  Please send all BUG REPORTS to:
 *  
 *  geom@cs.unc.edu
 *  
 *  The authors may be contacted via:
 *  
 *  Christian Lauterbach, Qi Mo, Jia Pan and Dinesh Manocha
 *  Dept. of Computer Science
 *  Frederick P. Brooks Jr. Computer Science Bldg.
 *  3175 University of N.C.
 *  Chapel Hill, N.C. 27599-3175
 *  United States of America
 *  
 *  http://gamma.cs.unc.edu/GPUCOL/
 *  
 */

#ifndef __CUDA_TIMER_H_
#define __CUDA_TIMER_H_

#ifdef _WIN32
//#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#ifndef _WIN32
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#endif

#include <cutil/cutil.h>

#include <cuda_runtime.h>

void startTimer(unsigned *timer);

double endTimer(char *info, unsigned *timer);

typedef struct TimerValue_t
{
#ifdef _WIN32
	void start()
	{
		QueryPerformanceCounter(&value);
		elapsed.QuadPart = 0;
	}
	
	void stop()
	{
		cudaThreadSynchronize();
		LARGE_INTEGER temp;
		QueryPerformanceCounter(&temp);
		elapsed.QuadPart = temp.QuadPart - value.QuadPart;
	}
	
	double getElapsed()
	{
		LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency);
		return (double)(elapsed.QuadPart) / (double)frequency.QuadPart;
	}
	
	double getElapsedMs()
	{
		LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency);
		return 1000.0 * ((double)elapsed.QuadPart / (double)frequency.QuadPart);
	}

    double getElapsedMicroSec()
    {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        return 1000000.0 * ((double)elapsed.QuadPart / (double)frequency.QuadPart);
    }
	
	LARGE_INTEGER value;
	LARGE_INTEGER elapsed;
#else
        void start()
        {
            struct timeval time;
            int status = gettimeofday(&time, NULL);
            value = time.tv_usec + time.tv_sec;
            // printf("start(): status %i, %li + %li = %li\n", status, time.tv_usec, time.tv_sec, value);
        }

        void stop()
        {
            cudaThreadSynchronize();
            int64_t temp;
            struct timeval time;
            /*int status = */gettimeofday(&time, NULL);
            temp = time.tv_usec + time.tv_sec;

            // printf("stop(): status %i, %li + %li = %li\n", status, time.tv_usec, time.tv_sec, temp);
            elapsed = temp - value;
            // printf("elapsed: %li - %li = %li\n", temp, value, elapsed);
        }

        double getElapsed()
        {
            /*int64_t frequency;
            struct timespec time_spec;
            int status = clock_getres(CLOCK_REALTIME, &time_spec);*/

            //double frequency = ((double)time_spec.tv_sec * 1000000.0) + ((double) time_spec.tv_nsec / 1000.0);

            /*printf("status = %i, time spec: %li -- %li, frequency: %f\n", status, time_spec.tv_sec, time_spec.tv_nsec, frequency);
            if (status == -1)
                printf("error %i: %s\n", errno, strerror(errno));*/

            return ((double)(elapsed) / 1000000 /*/ frequency*/);
        }

        double getElapsedMs()
        {
            /*int64_t frequency;
            struct timespec time_spec;
            clock_getres(CLOCK_REALTIME, &time_spec);*/

            // double frequency = ((double)time_spec.tv_sec * 1000000.0) + ((double) time_spec.tv_nsec / 1000.0);
            // frequency = (time_spec.tv_sec * 1000000) + (time_spec.tv_nsec / 1000);

            return ((double)(elapsed) / 1000 /* frequency*/);
        }

        double getElapsedMicroSec()
        {
            return ((double) elapsed);
        }

        int64_t value;
        int64_t elapsed;
#endif
} TimerValue;

#endif
