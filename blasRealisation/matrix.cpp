#include "matrix.hpp"
#include <xmmintrin.h>
#include <cstring>
#include <cblas.h>

#include <limits>


void Matrix::print() const {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      cout << mat[i][j] << ' ';
    }
    cout << '\n';
  }
  cout << std::flush;
}

int Matrix::getDim() const {
  return N;
}

float Matrix::getElem(const int& i, const int& j) const {
  if (i >= N || j >= N)
    throw std::length_error("out of bounds index");
  return mat[i][j];
}

void Matrix::setElem(const float& newVal, const int& i, const int& j) {
  if (i >= N || j >= N)
    throw std::length_error("out of bounds index");
  mat[i][j] = newVal;
}

float** Matrix::getMat() const {
  return mat;
}

float** Matrix::getMatCopy() const {
  float** matCopy = new float*[N];
  for (int i = 0; i < N; i++) {
    matCopy[i] = new float[N];
    for (int j = 0; j < N; j++)
      std::memcpy(matCopy[i], mat[i], N * sizeof(float));
  }
  return matCopy;
}

Matrix Matrix::operator-(const Matrix& A) {
  if (A.getDim() != this->getDim())
    throw std::invalid_argument("Matrix dimensions aren't the same");
  const int N = A.getDim();
  float **matA = A.getMat();
  float **thisMat = this->getMat();
  float **mat = new float*[N];
  for (int i = 0 ; i < N; i++) {
    mat[i] = new float[N]();
    cblas_saxpy(N, -1.0, matA[i], 1, mat[i], 1);
    cblas_saxpy(N, 1.0, thisMat[i], 1, mat[i], 1);
  }
  return Matrix(mat, N);
}

Matrix Matrix::operator+(const Matrix& A) {
   if (A.getDim() != this->getDim())
    throw std::invalid_argument("Matrix dimensions aren't the same");
  const int N = A.getDim();
  float **matA = A.getMat();
  float **thisMat = this->getMat();
  float **mat = new float*[N];
  for (int i = 0 ; i < N; i++) {
    mat[i] = new float[N]();
    cblas_saxpy(N, 1.0, matA[i], 1, mat[i], 1);
    cblas_saxpy(N, 1.0, thisMat[i], 1, mat[i], 1);
  }
  return Matrix(mat, N);
}

Matrix Matrix::operator*(const Matrix& A) {
  if (A.getDim() != this->getDim())
    throw std::invalid_argument("Matrix dimensions aren't the same");
  const int N = A.getDim();
  float *mat = new float[N * N];
  float *matA = new float[N * N];
  float *thisMat = new float[N * N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matA[i * N + j] = A.getElem(i, j);
      thisMat[i * N + j] = this->getElem(i, j);
    }
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, N, N, 1.0,
              thisMat, N, matA, N,
              0, mat, N);  
  Matrix Res(N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Res.setElem(mat[i * N + j], i, j);
    }
  }
  delete[] mat;
  delete[] matA;
  delete[] thisMat;
  return Res;
}

Matrix& Matrix::operator+=(const Matrix& A) {
  *this = *this + A;
  return *this;
}

Matrix& Matrix::operator-=(const Matrix& A) {
  *this = *this - A;
  return *this;
}

Matrix& Matrix::operator*=(const Matrix& A) {
  *this = *this * A;
  return *this;
}

Matrix& Matrix::operator=(const Matrix& A) {
  this->freeMat();
  this->mat = A.getMatCopy();
  return *this;
}

Matrix Matrix::Pow(const int& exp) const {
  if (exp <= 0)
    throw std::invalid_argument("Exponent must be > 0");
  Matrix A(*this);
  for (int i = 0; i < exp - 1; i++) {
    A = A * A;
  }
  return A;
}

void Matrix::readMatrix() {
  for (int i = 0; i < this->getDim(); i++) {
    for (int j = 0; j < this->getDim(); j++) {
      float x;
      cin >> x;
      this->setElem(x, i, j);
    }
  }
}

Matrix Matrix::transpose() {
  Matrix tMat(this->getDim());
  for (int i = 0; i < this->getDim(); i++) {
    for (int j = 0; j < this->getDim(); j++) {
      tMat.setElem(this->getElem(j, i), i, j);
    }
  }
  return tMat;
}

namespace {
  float Abs(float x) {
    return x < 0 ? -x : x;
  }
  float Max(float *vec, const int& N) {
    if (vec == nullptr)
      throw std::invalid_argument("vector is a nullptr");
    float max = vec[0];
    for (int i = 1; i < N; i++){
      max = max < vec[i] ? vec[i] : max;
    }
    return max;
  }
  void getA1Ainf(const Matrix& A, float* A1, float* Ainf) {
    const int N = A.getDim();
    float *A1sums = new float[N];
    *Ainf = std::numeric_limits<float>::min();
    for(int i = 0; i < N; i++) {
      float AinfTempSum = 0;
      for (int j = 0; j < N; j++) {
        if (i == 0)
          A1sums[j] = 0;
        A1sums[j] += Abs(A.getElem(i, j));
        AinfTempSum += Abs(A.getElem(i, j));
      }
      *Ainf = *Ainf < AinfTempSum ? AinfTempSum : *Ainf;
    }
    *A1 = Max(A1sums, N);
    delete[] A1sums;
  }
  Matrix getBMatrix(const Matrix& A, const float& A1, const float& Ainf) {
    const int N = A.getDim();
    Matrix B(N);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        B.setElem(A.getElem(j, i) / (Ainf * A1), i, j);
      }
    }
    return B;
  }
  
}

Matrix Matrix::invert(const int& exp) {
  const int N = this->getDim();
  Matrix I(N);
  I.IdentityMatrix();
  Matrix Res(N);
  Res.IdentityMatrix();
  float A1, Ainf;
  getA1Ainf(*this, &A1, &Ainf);
  Matrix B = getBMatrix(*this, A1, Ainf);
  Matrix BA = B * *this;
  Matrix R = I - BA;
  Matrix powR = R;
  for (int i = 1; i <= exp; i++) {
    Res += powR;
    if (i + 1 <= exp)
      powR *= R;
  }
  Res *= B;
  return Res;
}
