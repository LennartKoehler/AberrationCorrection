#include "aberration_backend/ZernikeCoefficients.h"


bool ZernikeCoefficients::anyLarger(const ZernikeCoefficients& other){
    for (int i = 0; i < Size; i++){
        if (c[i] > other.c[i]) return true;
    }
    return false;
}

ZernikeCoefficients ZernikeCoefficients::operator-(float value){
    ZernikeCoefficients coeff = *this;
    for (int i = 0; i < Size; i++){
        coeff.c[i] -= value;
    }
    return coeff;
}

ZernikeCoefficients ZernikeCoefficients::operator+(float value){
    ZernikeCoefficients coeff = *this;
    for (int i = 0; i < Size; i++){
        coeff.c[i] += value;
    }
    return coeff;
}

ZernikeCoefficients ZernikeCoefficients::operator-(const ZernikeCoefficients& other){
    ZernikeCoefficients coeff = *this;
    for (int i = 0; i < Size; i++){
        coeff.c[i] = c[i] - other.c[i];
    }
    return coeff;
}

ZernikeCoefficients ZernikeCoefficients::operator+(const ZernikeCoefficients& other){
    ZernikeCoefficients coeff = *this;
    for (int i = 0; i < Size; i++){
        coeff.c[i] = c[i] + other.c[i];
    }
    return coeff;
}
void ZernikeCoefficients::clamp(const ZernikeCoefficients& other){
    for (int i = 0; i < Size; i++){
        if (c[i] > other.c[i]){
            c[i] = other.c[i];
        }
    }
}

void ZernikeCoefficients::floor(const ZernikeCoefficients& other){
    for (int i = 0; i < Size; i++){
        if (c[i] < other.c[i]){
            c[i] = other.c[i];
        }
    }
}
