#include <iostream>
#include "matrix.hpp"

using namespace std;

int main() {
  int N = 4;
  cin >> N;
  float **matA = new float*[N];
  for (int i = 0; i < N; i++) {
    matA[i] = new float[N];
  }
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matA[i][j] = rand() % 1000;
    }
  }
  Matrix A(matA, N);
  A.print();
  cout << endl;
  Matrix R = A.invert(1000);
  cout << "Inverted:" << endl;
  R.print();
  cout << endl;
  Matrix mul = A * R;
  cout << "Result:" << endl;
  mul.print();
  return 0;
}