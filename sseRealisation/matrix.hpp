#ifndef _FLOATMATRIX_MATRIX_HPP
#define _FLOATMATRIX_MATRIX_HPP

// check _mm_rcp_ps and div and abs

#include <iostream>
#include <vector>
#include <algorithm>

using std::vector;
using std::cin;
using std::cout;
using std::endl;

class Matrix {
  int N;
  float **mat;
  void freeMat() {
    for (int i = 0; i < N; i++)
      delete[] mat[i];
    delete[] mat;
  }
  public:
    virtual ~Matrix() {
      freeMat();
    }
    Matrix(const int &n) {
      N = n;
      mat = new float*[N];
      for (int i = 0; i < N; i++) {
        mat[i] = new float[N];
        std::fill(mat[i], mat[i] + N, 0);
      }
    }
    Matrix(float** matrix, const int& n) {
      N = n;
      mat = matrix;
    }
    Matrix(const Matrix& A) {
      N = A.getDim();
      mat = A.getMatCopy();
    }
    Matrix& IdentityMatrix() {
      for (int i = 0; i < N; i++) {
        std::fill(this->mat[i], this->mat[i] + N, 0);
        setElem(1.0, i, i);
      }
      return *this;
    }
    void print() const;
    void readMatrix();
    int getDim() const;
    float getElem(const int& i, const int& j) const;
    void setElem(const float& newVal, const int& i, const int& j);
    float** getMat() const;
    float** getMatCopy() const;
 
    Matrix transpose();
    Matrix invert(const int& exp);

    Matrix operator-(const Matrix& A);
    Matrix operator+(const Matrix& A);
    Matrix operator*(const Matrix& A);
    Matrix Pow(const int& exp) const;

    Matrix& operator=(const Matrix& A);
    Matrix& operator-=(const Matrix& A);
    Matrix& operator+=(const Matrix& A);
    Matrix& operator*=(const Matrix& A);
};

// Matrix invert(const Matrix& A, const int& exp);

#endif  // _FLOATMATRIX_MATRIX_HPP