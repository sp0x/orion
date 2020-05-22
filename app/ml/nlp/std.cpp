#include "std.h"

double sum(double *arr, size_t sz){
    double out  = 0;
    for(int i=0; i<sz; i++){
        out += arr[i];
    }
    return out;
}
