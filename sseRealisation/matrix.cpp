#include "matrix.hpp"
#include "intrinsics.hpp"
#include <xmmintrin.h>
#include <cstring>

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
  float **mat = new float*[N];
  float **thisMat = this->getMat();
  float **AMat = A.getMat();
  for (int i = 0 ; i < N; i++) {
    mat[i] = intrinsics::subtVectors(thisMat[i], AMat[i], N);
  }
  return Matrix(mat, N);
}

Matrix Matrix::operator+(const Matrix& A) {
  if (A.getDim() != this->getDim())
    throw std::invalid_argument("Matrix dimensions aren't the same");
  const int N = A.getDim();
  float **mat = new float*[N];
  float **thisMat = this->getMat();
  float **AMat = A.getMat();
  __m128 *sums = (__m128*)_mm_malloc(N * sizeof(float), 16);
  for (int i = 0 ; i < N; i++) {
    mat[i] = intrinsics::sumVectors(thisMat[i], AMat[i], N);
  }
  _mm_free(sums);
  return Matrix(mat, N);
}

Matrix Matrix::operator*(const Matrix& A) {
  if (A.getDim() != this->getDim())
    throw std::invalid_argument("Matrix dimensions aren't the same");
  const int N = A.getDim();
  Matrix res(N);
  float **mat = new float*[N];
  std::cout << "MULT 1" << std::endl;
  float **thisMat = this->getMat();
  float **AMat = A.getMat();
  float *temp = new float[N];
  std::cout << "MULT 2" << std::endl;
  for (int i = 0; i < N; i++) {
    mat[i] = intrinsics::mulVectorByMatrix(thisMat[i], AMat, N);
    std::cout << "MULT " << 3 + i << " " << std::endl;
  }
  std::cout << "MULT 3" << std::endl;
  delete[] temp;
  return Matrix(mat, N);
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

// namespace {
//   float** getBMatrix(float **A, const int& N, const float& A1, const float& Ainf) {
//     std::cout<<"Here 5"<<std::endl;
//     float** B = new float*[8];
//     std::cout<<"Here 4"<<std::endl;
//     for (int i = 0; i < N; i++) {
//       B[i] = new float[N];
//       // for (int j = 0; j < N; j++) {
//       //   B[i][j] = A[j][i] / (Ainf * A1);
//       // }
//     }
//     return B;
//   }
// }

void getBMatrix(float **A, float **B, const int& N, const float& A1, const float& Ainf) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      B[i][j] = A[j][i] / (Ainf * A1);
    }
  }
}

Matrix invert(const Matrix& A, const int& exp) {
  const int N = A.getDim();
  Matrix I(N);
  I.IdentityMatrix();
  Matrix Res(N);
  Res.IdentityMatrix();
  float **matA = A.getMat();
  // float **BMat = intrinsics::getBMatrix(thisMat, intrinsics::getA1(thisMat), intrinsics::getAinf(thisMat));
  float **matB = new float*[N];
  for (int i = 0; i < N; i++)
    matB[i] = new float[N];
  getBMatrix(matA, matB, N, intrinsics::getA1(matA, N), intrinsics::getAinf(matA, N));
  std::cout << "Here" << std::endl;
  Matrix B(matB, N);
  std::cout << "Here 1" << std::endl;
  Matrix R = I - B * A;
  std::cout << "Here 2" << std::endl;
  for (int i = 1; i <= exp; i++) {
    Res += R;
    if (i + 1 <= exp)
      R *= R;
  }
  Res *= B;
  return Res;
}

int main() {
  int N;
  cin >> N;
  float **matA = new float*[N];
  float **matB = new float*[N];
  for (int i = 0; i < N; i++){
    matA[i] = new float[N];
    matB[i] = new float[N];
  }
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matA[i][j] = rand() % 10;
      matB[i][j] = rand() % 10;
    }
  }
  // float **BMat = getBMatrix(matA, N, 1, 1);
  Matrix A(matA, N);
  A.print();
  cout << endl;
  Matrix B(matB, N);
  B.print();
  cout << endl;
  Matrix mult = A * B;
  mult.print();
  return 0;
}