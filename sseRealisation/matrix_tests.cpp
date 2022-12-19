#include "gtest/gtest.h"
#include "matrix.hpp"

TEST(matrixOperations, subtractionOperator) {
  const int N = 6;
  float **matA = new float*[N];
  float **matB = new float*[N];
  for (int i = 0; i < N; i++) {
    matA[i] = new float[N];
    matB[i] = new float[N];
    for (int j = 0; j < N; j++) {
      matA[i][j] = i * N + j + 1;
      matB[i][j] = 2 * i * N + 2 * j + 2;
    }
  }
  Matrix A(matA, N);
  Matrix B(matB, N);
  Matrix Res = A - B;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      EXPECT_EQ(Res.getElem(i, j), A.getElem(i, j) - B.getElem(i, j)) <<
        "At index: (" << i << ", " << j << ")";
    }
  }
}

TEST(matrixOperations, additionOperator) {
  const int N = 6;
  float **matA = new float*[N];
  float **matB = new float*[N];
  for (int i = 0; i < N; i++) {
    matA[i] = new float[N];
    matB[i] = new float[N];
    for (int j = 0; j < N; j++) {
      matA[i][j] = i * N + j + 1;
      matB[i][j] = 2 * i * N + 2 * j + 2;
    }
  }
  Matrix A(matA, N);
  Matrix B(matB, N);
  Matrix Res = A + B;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      EXPECT_EQ(Res.getElem(i, j), A.getElem(i, j) + B.getElem(i, j)) <<
        "At index: (" << i << ", " << j << ")";
    }
  }
}

TEST(matrixOperations, multiplicationOperator) {
  const int N = 6;
  float **matA = new float*[N];
  float **matB = new float*[N];
  for (int i = 0; i < N; i++) {
    matA[i] = new float[N];
    matB[i] = new float[N];
    for (int j = 0; j < N; j++) {
      matA[i][j] = i * N + j + 1;
      matB[i][j] = 2 * i * N + 2 * j + 2;
    }
  }

  // A =
  // 1 2 3 4 5 6
  // 7 8 9 10 11 12
  // 13 14 15 16 17 18
  // 19 20 21 22 23 24
  // 25 26 27 28 29 30
  // 31 32 33 34 35 36

  // B = 
  // 2 4 6 8 10 12
  // 14 16 18 20 22 24
  // 26 28 30 32 34 36
  // 38 40 42 44 46 48
  // 50 52 54 56 58 60
  // 62 64 66 68 70 72

  // A * B =
  // 882   924   966  1008  1050  1092
  // 2034  2148  2262  2376  2490  2604
  // 3186  3372  3558  3744  3930  4116
  // 4338  4596  4854  5112  5370  5628
  // 5490  5820  6150  6480  6810  7140
  // 6642  7044  7446  7848  8250  8652


  Matrix A(matA, N);
  Matrix B(matB, N);
  Matrix Res = A * B;

  float expected[N][N] {
    882,   924,   966,  1008,  1050,  1092,
    2034,  2148,  2262,  2376,  2490,  2604,
    3186,  3372,  3558,  3744,  3930,  4116,
    4338,  4596,  4854,  5112,  5370,  5628,
    5490,  5820,  6150,  6480,  6810,  7140,
    6642,  7044,  7446,  7848,  8250,  8652
  };
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      EXPECT_EQ(Res.getElem(i, j), expected[i][j]) <<
        "At index: (" << i << ", " << j << ")";
    }
  }
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}