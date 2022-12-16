#include "gtest/gtest.h"
#include "intrinsics.hpp"
#include <xmmintrin.h>
#include <iostream>

TEST(IntrinsicsTest, sum128vector) {
  const int N = 4;
  float *vec = new float[N]{1.4, 2.6, 3.5, 1.7};
  float sum = 0;
  for (int i = 0; i < N; i++) {
    sum += vec[i];
  }
  __m128 v = _mm_loadu_ps(vec);
  float m128sum = intrinsics::sum128vector(v);
  std::cout << std::fixed << std::setprecision(6) << m128sum << ' ' << sum << std::endl;
  EXPECT_EQ(m128sum, sum);
  delete[] vec;
}

TEST(IntrinsicsTest, sumVectors) {
  const int N = 5;
  float *v1 = new float[N]{1.4, 2.6, 3.5, 1.7, 1.8};
  float *v2 = new float[N]{1.6, 3.4, 3.5, -1.1, 0};
  float *sum = intrinsics::sumVectors(v1, v2, N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(sum[i], v1[i] + v2[i]) << "at index: " << i;
  }
  delete[] v1;
  delete[] v2;
  delete[] sum;
}

TEST(IntrinsicsTest, subtVectors) {
  const int N = 5;
  float *v1 = new float[N]{1.4, 2.6, 3.5, 1.7, 1.8};
  float *v2 = new float[N]{1.6, 3.4, 3.5, -1.1, 0};
  float *sum = intrinsics::subtVectors(v1, v2, N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(sum[i], v1[i] - v2[i]) << "at index: " << i;
  }
  delete[] v1;
  delete[] v2;
  delete[] sum;
}

TEST(IntrinsicsTest, mulVectors) {
  const int N = 5;
  float *v1 = new float[N]{1.4, 2.6, 3.5, 1.7, 1.8};
  float *v2 = new float[N]{1.6, 3.4, 3.5, -1.1, 0};
  float *sum = intrinsics::mulVectors(v1, v2, N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(sum[i], v1[i] * v2[i]) << "at index: " << i;
  }
  delete[] v1;
  delete[] v2;
  delete[] sum;
}

TEST(IntrinsicsTest, mulAndSumVectors) {
  const int N = 5;
  float *v1 = new float[N]{1.4, 2.6, 3.5, 1.7, 1.8};
  float *v2 = new float[N]{1.6, 3.4, 3.5, -1.1, 0};
  float sum = intrinsics::mulAndSumVectors(v1, v2, N);
  float expected = 0;
  for (int i = 0; i < N; i++) {
    expected += v1[i] * v2[i];
  }
  EXPECT_EQ(sum, expected);
  delete[] v1;
  delete[] v2;
}

TEST(IntrinsicsTest, getMaxValue) {
  const int N = 4;
  float *v1 = new float[N]{1.4, 2.6, 3.5, 1.7};
  float *v2 = new float[N]{1.6, 3.4, 3.6, -1.1};
  float *v3 = new float[N]{2.6, 4.05, 3, 0.5};
  __m128 *vec = (__m128*)_mm_malloc(3 * N * sizeof(float), 16);
  vec[0] = _mm_loadu_ps(v2);
  vec[1] = _mm_loadu_ps(v1);
  vec[2] = _mm_loadu_ps(v3);
  float maxValue = intrinsics::getMaxValue(vec, 3);
  float expected = 4.05;
  EXPECT_EQ(maxValue, expected);
  delete[] v1;
  delete[] v2;
  delete[] v3;
  _mm_free(vec);
}

TEST(IntrinsicsTest, getA1) {
  const int N = 5;
  float **mat = new float*[N];
  for (int i = 0; i < N; i++) {
    mat[i] = new float[N];
    for (int j = 0; j < N; j++) {
      mat[i][j] = i * N + j + 1;
    }
  }

  // 1 2 3 4 5
  // 6 7 8 9 10
  // 11 12 13 14 15
  // 16 17 18 19 20
  // 21 22 23 24 25

  float A1 = intrinsics::getA1(mat, N);
  float expected = 75;
  EXPECT_EQ(A1, expected);
  for (int i = 0; i < N; i++) {
    delete[] mat[i];
  }
  delete[] mat;
}

TEST(IntrinsicsTest, getAinf) {
  const int N = 5;
  float **mat = new float*[N];
  for (int i = 0; i < N; i++) {
    mat[i] = new float[N];
    for (int j = 0; j < N; j++) {
      mat[i][j] = i * N + j + 1;
    }
  }

  // 1 2 3 4 5
  // 6 7 8 9 10
  // 11 12 13 14 15
  // 16 17 18 19 20
  // 21 22 23 24 25

  float A1 = intrinsics::getAinf(mat, N);
  float expected = 115;
  EXPECT_EQ(A1, expected);
  for (int i = 0; i < N; i++) {
    delete[] mat[i];
  }
  delete[] mat;
}

TEST(IntrinsicsTest, mulVectorByMatrix) {
  const int N = 5;
  float **mat = new float*[N];
  for (int i = 0; i < N; i++) {
    mat[i] = new float[N];
    for (int j = 0; j < N; j++) {
      mat[i][j] = i * N + j + 1;
    }
  }

  // 1 2 3 4 5
  // 6 7 8 9 10
  // 11 12 13 14 15
  // 16 17 18 19 20
  // 21 22 23 24 25

  float *vec = new float[N]{3, 2, 4, 5, 7};
  float *expected = new float[N]();

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      expected[j] += vec[i] * mat[i][j];
    }
  }
  float *res = intrinsics::mulVectorByMatrix(vec, mat, N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(res[i], expected[i]) << "at index: " << i;
  }
  
  for (int i = 0; i < N; i++) {
    delete[] mat[i];
  }
  delete[] mat;
  delete[] vec;
  delete[] res;
  delete[] expected;
}

TEST(IntrinsicsTest, mulTransposedByItself) {
  const int N = 5;
  float **mat = new float*[N];
  for (int i = 0; i < N; i++) {
    mat[i] = new float[N];
    for (int j = 0; j < N; j++) {
      mat[i][j] = i * N + j + 1;
    }
  }

  // A = 
  // 1 2 3 4 5
  // 6 7 8 9 10
  // 11 12 13 14 15
  // 16 17 18 19 20
  // 21 22 23 24 25

  // At * A = 
  // 855	 910	 965	1020	1075
  // 910	 970	1030	1090	1150
  // 965	1030	1095	1160	1225
  // 1020	1090	1160	1230	1300
  // 1075	1150	1225	1300	1375


  float expected[N][N] =
    { 855, 910, 965, 1020, 1075,
      910, 970, 1030, 1090, 1150,
      965, 1030, 1095,	1160, 1225,
      1020, 1090, 1160, 1230, 1300,
      1075, 1150, 1225, 1300, 1375 };

  float **BA = intrinsics::getBAMatrix(mat, N, 1, 1);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      EXPECT_EQ(BA[i][j], expected[i][j]) <<
        "at index: (" << i << ", " << j << ")";
    }
  }
  
  for (int i = 0; i < N; i++) {
    delete[] mat[i];
    delete[] BA[i];
  }
  delete[] mat;
  delete[] BA;
}

TEST(IntrinsicsTest, getBAMatrix) {
  const int N = 5;
  float **mat = new float*[N];
  for (int i = 0; i < N; i++) {
    mat[i] = new float[N];
    for (int j = 0; j < N; j++) {
      mat[i][j] = i * N + j + 1;
    }
  }

  // A = 
  // 1 2 3 4 5
  // 6 7 8 9 10
  // 11 12 13 14 15
  // 16 17 18 19 20
  // 21 22 23 24 25

  // At * A = 
  // 855	 910	 965	1020	1075
  // 910	 970	1030	1090	1150
  // 965	1030	1095	1160	1225
  // 1020	1090	1160	1230	1300
  // 1075	1150	1225	1300	1375


  float expected[N][N] =
    { 855, 910, 965, 1020, 1075,
      910, 970, 1030, 1090, 1150,
      965, 1030, 1095,	1160, 1225,
      1020, 1090, 1160, 1230, 1300,
      1075, 1150, 1225, 1300, 1375 };

  float A1 = intrinsics::getA1(mat, N);
  float Ainf = intrinsics::getAinf(mat, N);

  float **BA = intrinsics::getBAMatrix(mat, N, A1, Ainf);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      EXPECT_EQ(BA[i][j], expected[i][j] / (A1 * Ainf)) <<
        "at index: (" << i << ", " << j << ")";
    }
  }
  
  for (int i = 0; i < N; i++) {
    delete[] mat[i];
    delete[] BA[i];  
  }
  delete[] mat;
  delete[] BA;
}

TEST(IntrinsicsTest, mulMatByTransposed) {
  const int N = 5;
  float **matA = new float*[N];
  float **matB = new float*[N];
  for (int i = 0; i < N; i++) {
    matA[i] = new float[N];
    matB[i] = new float[N];
    for (int j = 0; j < N; j++) {
      matA[i][j] = i * N + j + 1;
      matB[i][j] = i * N + j + 1;
    }
  }

  // A = 
  // 1 2 3 4 5
  // 6 7 8 9 10
  // 11 12 13 14 15
  // 16 17 18 19 20
  // 21 22 23 24 25

  // A * At = 
  // 55	130	 205	 280	 355
  // 130	330	 530	 730	 930
  // 205	530	 855	1180	1505
  // 280	730	1180	1630	2080
  // 355	930	1505	2080	2655



  float expected[N][N] =
    { 55, 130, 205, 280, 355,
      130, 330, 530, 730, 930,
      205, 530, 855, 1180, 1505,
      280, 730, 1180, 1630, 2080,
      355, 930, 1505, 2080, 2655 };

  float **Res = intrinsics::mulMatByTransposed(matA, matB, N, 1, 1);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      EXPECT_EQ(Res[i][j], expected[i][j]) <<
        "at index: (" << i << ", " << j << ")";
    }
  }
  
  for (int i = 0; i < N; i++) {
    delete[] matA[i];
    delete[] matB[i];
    delete[] Res[i];
  }
  delete[] matA;
  delete[] matB;
  delete[] Res;
}

TEST(IntrinsicsTest, mulMatByTransposedWithScalars) {
  const int N = 5;
  float **matA = new float*[N];
  float **matB = new float*[N];
  for (int i = 0; i < N; i++) {
    matA[i] = new float[N];
    matB[i] = new float[N];
    for (int j = 0; j < N; j++) {
      matA[i][j] = i * N + j + 1;
      matB[i][j] = i * N + j + 1;
    }
  }

  // A = 
  // 1 2 3 4 5
  // 6 7 8 9 10
  // 11 12 13 14 15
  // 16 17 18 19 20
  // 21 22 23 24 25

  // A * At = 
  // 55	130	 205	 280	 355
  // 130	330	 530	 730	 930
  // 205	530	 855	1180	1505
  // 280	730	1180	1630	2080
  // 355	930	1505	2080	2655



  float expected[N][N] =
    { 55, 130, 205, 280, 355,
      130, 330, 530, 730, 930,
      205, 530, 855, 1180, 1505,
      280, 730, 1180, 1630, 2080,
      355, 930, 1505, 2080, 2655 };

  float A1 = intrinsics::getA1(matA, N);
  float Ainf = intrinsics::getAinf(matA, N);

  float **Res = intrinsics::mulMatByTransposed(matA, matB, N, A1, Ainf);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      EXPECT_EQ(Res[i][j], expected[i][j] / (A1 * Ainf)) <<
        "at index: (" << i << ", " << j << ")";
    }
  }
  
  for (int i = 0; i < N; i++) {
    delete[] matA[i];
    delete[] matB[i];
    delete[] Res[i];
  }
  delete[] matA;
  delete[] matB;
  delete[] Res;
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
