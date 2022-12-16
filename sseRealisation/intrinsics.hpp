#ifndef _FLOATMATRIX_INTRINSICS_HPP
#define _FLOATMATRIX_INTRINSICS_HPP

#include <xmmintrin.h>

namespace intrinsics {
  __m128 padVectorWithZeros(float* vec, int padding) {
    switch (padding) {
      case 0: return _mm_set_ps(vec[3], vec[2], vec[1], vec[0]);
      case 1: return _mm_set_ps(0, vec[2], vec[1], vec[0]);
      case 2: return _mm_set_ps(0, 0, vec[1], vec[0]);
      case 3: return _mm_set_ps(0, 0, 0, vec[0]);
      default: return _mm_set_ps(0, 0, 0, 0);
    }
  }

  float sum128vector(__m128 vec) {
    __m128 temp = _mm_movehl_ps(vec, vec);
    vec = _mm_add_ps(vec, temp);
    temp = _mm_shuffle_ps(vec, vec, 1);
    vec = _mm_add_ps(vec, temp);
    float sum;
    _mm_store_ss(&sum, vec);
    return sum;
  }

  float* sumVectors(float* v1, float *v2, const int& N) {
    float *res = new float[N];
    __m128 tempSum;
    for (int i = 0; i < N / 4; i++) {
      __m128 vec1 = _mm_loadu_ps(&v1[i * 4]);
      __m128 vec2 = _mm_loadu_ps(&v2[i * 4]);
      tempSum = _mm_add_ps(vec1, vec2);
      _mm_storeu_ps(&res[i * 4], tempSum);
    }
    if (N % 4 != 0) {
      int idx = N / 4;
      idx *= 4;
      while (idx < N) {
        res[idx] = v1[idx] + v2[idx];
        idx++;
      }
    }
    return res;
  }

  float* subtVectors(float* v1, float *v2, const int& N) {
    float *res = new float[N];
    __m128 tempSum;
    for (int i = 0; i < N / 4; i++) {
      __m128 vec1 = _mm_loadu_ps(&v1[i * 4]);
      __m128 vec2 = _mm_loadu_ps(&v2[i * 4]);
      tempSum = _mm_sub_ps(vec1, vec2);
      _mm_storeu_ps(&res[i * 4], tempSum);
    }
    if (N % 4 != 0) {
      int idx = N / 4;
      idx *= 4;
      while (idx < N) {
        res[idx] = v1[idx] - v2[idx];
        idx++;
      }
    }
    return res;
  }

  float* mulVectors(float* v1, float *v2, const int& N) {
    float *res = new float[N];
    __m128 tempSum;
    for (int i = 0; i < N / 4; i++) {
      __m128 vec1 = _mm_loadu_ps(&v1[i * 4]);
      __m128 vec2 = _mm_loadu_ps(&v2[i * 4]);
      tempSum = _mm_mul_ps(vec1, vec2);
      _mm_storeu_ps(&res[i * 4], tempSum);
    }
    if (N % 4 != 0) {
      int idx = N / 4;
      idx *= 4;
      while (idx < N) {
        res[idx] = v1[idx] * v2[idx];
        idx++;
      }
    }
    return res;
  }

  float mulAndSumVectors(float *v1, float *v2, const int& N) {
    const int sumsLen = N / 4;
    __m128 *sums = (__m128*)_mm_malloc(sumsLen * sizeof(float), 16);
    sums[0] = _mm_setzero_ps();
    for (int i = 0; i < N / 4; i++) {
      __m128 vec1 = _mm_loadu_ps(&v1[i * 4]);
      __m128 vec2 = _mm_loadu_ps(&v2[i * 4]);
      sums[i] = _mm_mul_ps(vec1, vec2);
      if (i != 0)
        sums[0] = _mm_add_ps(sums[0], sums[i]);
    }
    float res = sum128vector(sums[0]);
    if (N % 4 != 0) {
      int idx = N / 4;
      idx *= 4;
      while (idx < N) {
        res += v1[idx] * v2[idx];
        idx++;
      }
    }
    _mm_free(sums);
    return res;
  }

  __m128 storeSumInFirstElem(__m128 vec) {
    __m128 temp = _mm_movehl_ps(vec, vec);
    vec = _mm_add_ps(vec, temp);
    temp = _mm_shuffle_ps(vec, vec, 1);
    vec = _mm_add_ps(vec, temp);
    return vec;
  }

  __m128 storeMaxInFirstElem(__m128 vec) {
    __m128 temp = _mm_movehl_ps(vec, vec);
    vec = _mm_max_ps(vec, temp);
    temp = _mm_shuffle_ps(vec, vec, 1);
    vec = _mm_max_ps(vec, temp);
    return vec;
  }

  float getMaxValue(__m128* vec, const int& len) {
    __m128 maxVec = _mm_setzero_ps();
    for (int i = 0; i < len; i++) {
      maxVec = _mm_max_ps(maxVec, vec[i]);
    }
    maxVec = storeMaxInFirstElem(maxVec);
    float res;
    _mm_store_ss(&res, maxVec);
    return res;
  }

  float getA1(float **mat, const int& N) {
    const int sumsLen = N / 4;
    float *leftovers = nullptr;
    const int leftoversLen = N - sumsLen * 4;
    if (leftoversLen != 0) {
      leftovers = new float[leftoversLen]();
    }
    __m128 *sums = (__m128*)_mm_malloc(sumsLen * 4 * sizeof(float), 16);
    for (int row = 0; row < N; row++) {
      for (int col = 0; col < N / 4; col++) {
        if (row == 0)
          sums[col] = _mm_setzero_ps();
        __m128 vec = _mm_loadu_ps(&mat[row][col * 4]);
        sums[col] = _mm_add_ps(sums[col], vec);
      }
      if (leftoversLen != 0) {
        int idx = N - leftoversLen;
        int i = 0;
        while (idx < N) {
          leftovers[i++] += mat[row][idx++];
        }
      }
    }
    float maxSum = getMaxValue(sums, sumsLen);
    for (int i = 0; i < leftoversLen; i++) {
      maxSum = maxSum < leftovers[i] ? leftovers[i] : maxSum;
    }
    _mm_free(sums);
    delete[] leftovers;
    return maxSum;
  }

  float getAinf(float **mat, const int& N) {
    float maxSum = std::numeric_limits<float>::min();
    for (int row = 0; row < N; row++) {
      __m128 sums = _mm_setzero_ps();
      for (int col = 0; col < N / 4; col++) {
        __m128 vec = _mm_loadu_ps(&mat[row][col * 4]);
        sums = _mm_add_ps(sums, vec);
      }
      float sum = sum128vector(sums);
      if (N % 4 != 0) {
        int idx = N / 4;
        idx *= 4;
        while (idx < N) {
          sum += mat[row][idx++];
        }
      }
      maxSum = maxSum < sum ? sum : maxSum;
    }
    return maxSum;
  }

  float* mulVectorByMatrix(float* vec, float** mat, const int& N) {
    float *res = new float[N];
    const int sumsLen = N / 4;
    const int leftoversLen = N - sumsLen * 4;
    __m128 *sums = (__m128*)_mm_malloc(sumsLen * 4 * sizeof(float), 16);
    __m128 temp;
    for (int i = 0; i < N; i++) {
      __m128 scalar = _mm_load1_ps(&vec[i]);
      for (int j = 0; j < sumsLen; j++) {
        if (i == 0)
          sums[j] = _mm_setzero_ps();
        temp = _mm_loadu_ps(&mat[i][j * 4]);
        temp = _mm_mul_ps(temp, scalar);
        sums[j] = _mm_add_ps(sums[j], temp);
        if (i == N - 1) {
          _mm_storeu_ps(&res[j * 4], sums[j]);
        }
      }
      if (leftoversLen != 0) {
        int matIdx = N - leftoversLen;
        while (matIdx < N) {
          if (i == 0) {
            res[matIdx] = 0;
          }
          res[matIdx] += vec[i] * mat[i][matIdx];
          matIdx++;
        }
      }
    }
    _mm_free(sums);
    return res;
  }

  float** getBAMatrix(float **A, const int& N, const float& A1, const float& Ainf) {
    float scalar = A1 * Ainf;
    __m128 scalarVec = _mm_load1_ps(&scalar);
    int sumsLen = N / 4;
    int leftoversLen = N - sumsLen * 4;
    __m128 *sums = (__m128*)_mm_malloc(sumsLen * 4 * sizeof(float), 16);
    float **mat = new float*[N];      
    for (int i = 0; i < N; i++) {
      mat[i] = new float[N];
      for (int j = 0; j < N; j++) {
        __m128 matrixEl = _mm_load1_ps(&A[j][i]);
        for (int k = 0; k < sumsLen; k++) {
          if (j == 0)
            sums[k] = _mm_setzero_ps();
          __m128 temp = _mm_load_ps(&A[j][k * 4]);
          temp = _mm_mul_ps(matrixEl, temp);
          sums[k] = _mm_add_ps(sums[k], temp);
          if (j == N - 1) {
            sums[k] = _mm_div_ps(sums[k], scalarVec);
            _mm_storeu_ps(&mat[i][k * 4], sums[k]);
          }
        }
        if (leftoversLen != 0) {
          int idx = N - leftoversLen;
          while (idx < N) {
            if (j == 0)
              mat[i][idx] = 0;
            mat[i][idx] += A[j][idx] * A[j][i];
            if (j == N - 1)
              mat[i][idx] /= scalar;
            idx++;
          }
        }
      }
    }
    _mm_free(sums);
    return mat;
  }

  //  TODO: MAKE FUNCTIONS VOID TYPE !!!

  // res = S * At
  float** mulMatByTransposed(float **S, float **A, const int& N,
                                   const float& A1, const float& Ainf) {
    float **res = new float*[N];
    for (int i = 0; i < N; i++) {
      res[i] = new float[N];
      float *v1 = S[i];
      for (int j = 0; j < N; j++) {
        float *v2 = A[j];
        res[i][j] = mulAndSumVectors(v1, v2, N) / (A1 * Ainf);
      }
    }
    return res;
  }
} // namespace intrinsics

#endif  // _FLOATMATRIX_INTRINSICS_HPP