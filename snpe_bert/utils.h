/*
 * @Description: Common header.
 * @version: 2.0
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-07-09 13:07:37
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2023-02-24 15:12:34
 */
#pragma once

#include <algorithm>
#include <functional>
#include <math.h>
#include <chrono>

#include "Logger.h"

#if defined(_MSC_VER)
  #ifdef DLL_EXPORTS
    #define EXPORT_API __declspec(dllexport)
  #else
    #define EXPORT_API __declspec(dllimport)
  #endif
#elif __GNUC__ >= 4
    #define EXPORT_API __attribute__ ((visibility("default")))
#else
    #define EXPORT_API
#endif

// Inference hardware runtime.
typedef enum runtime {
    CPU = 0,
    GPU,
    GPU_FLOAT16,
    DSP,
    DSP_FIXED8,
    AIP
}runtime_t;

static int64_t GetTimeStamp_ms()
{
    std::chrono::time_point<std::chrono::system_clock,std::chrono::milliseconds> tp = 
    std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    std::time_t timestamp =  tp.time_since_epoch().count();
    return timestamp;
}
