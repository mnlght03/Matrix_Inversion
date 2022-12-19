#include "matrix.hpp"
#include <chrono>

const int N = 128;
const int M = 1000;

Matrix getL() {
  float **matL = new float*[N];
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    matL[i] = new float[N];
    for (int j = 0; j < N; j++) {
      matL[i][j] = j <= i ? rand() % 100 : 0;
    }
  }
  return Matrix(matL, N);
}

Matrix getU() {
  float **matU = new float*[N];
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    matU[i] = new float[N];
    for (int j = 0; j < N; j++) {
      matU[i][j] = j >= i ? rand() % 100 : 0;
    }
  }
  return Matrix(matU, N);
}

int main() {
  Matrix L = getL();
  Matrix U = getU();
  Matrix A = L * U;
  auto start = std::chrono::steady_clock::now();
  A.invert(M);
  auto end = std::chrono::steady_clock::now();
  auto s = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Blas Realisation Elapsed Time: " << s << "s " << ms << "ms" << std::endl;
  return 0;
}