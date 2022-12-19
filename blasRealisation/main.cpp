#include <iostream>
#include <ctime>
#include "matrix.hpp"

int main() {
    int N;
    std::cin >> N;
    srand(time(NULL));
    float **mat = new float*[N];
    for (int i = 0; i < N; i++) {
        mat[i] = new float[N];
        for (int j = 0; j < N; j++) {
            mat[i][j] = rand() % 100;
        }
    }
    Matrix A(mat, N);
    std::cout << "A:" << std::endl;
    A.print();
    std::cout << std::endl;
    Matrix inversedA = A.invert(1000);
    std::cout << "Inversed:" << std::endl; 
    inversedA.print();
    std::cout << std::endl;
    Matrix mul = A * inversedA;
    mul.print();
    return 0;
}