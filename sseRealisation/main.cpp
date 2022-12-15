#include <iostream>
#include "matrix.hpp"

int main() {
    Matrix A();
    A.readMatrix();
    Matrix inversedA = A.invert();
    return 0;
}