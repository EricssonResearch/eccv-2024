#include "eripose/eripose.h"

#include <Eigen/Dense>
#include <chrono>
#include <iostream>

using namespace Eigen;
using namespace std;

int main() {
    /* Timing experiments */
    int nbr_iter = 100000;
    cout << "Running: " << nbr_iter << " times" << endl;
    Vector4d out;

    Matrix<double, 12, 1> data = Matrix<double, 12, 1>::Random(12, 1);
    Matrix4d A = data.reshaped(4, 3) * data.reshaped(4, 3).transpose();
    Matrix<double, 10, 1> data_mod;
    data_mod << A(0, 0), A(0, 1), A(0, 2), A(0, 3), A(1, 1), A(1, 2), A(1, 3), A(2, 2), A(2, 3), A(3, 3);

    // Guan et al. (CVPR 2020) - LS method
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < nbr_iter; i++) {
        out = EricssonResearch::solver_guan_cvpr_2020_ls(data);
    }
    auto end = chrono::steady_clock::now();
    cout << "Mean execution time (Guan et al. (LS)): "
         << chrono::duration_cast<chrono::nanoseconds>(end - start).count() / nbr_iter << " ns" << endl;

    // Valtonen Ornhag et al. (ECCV 2024)
    Matrix<double, 8, 1> data2 = Matrix<double, 8, 1>::Random();
    std::pair<Eigen::Matrix3d, Eigen::Matrix3d> out_sols;
    start = chrono::steady_clock::now();
    for (int i = 0; i < nbr_iter; i++) {
        out_sols = EricssonResearch::solver_valtonen_ornhag_eccv_2024(data2);
    }
    end = chrono::steady_clock::now();
    cout << "Mean execution time (solver_valtonen_ornhag_eccv_2024): "
         << chrono::duration_cast<chrono::nanoseconds>(end - start).count() / nbr_iter << " ns" << endl;

    // Choi and Kim (IVC 2018)
    std::vector<Eigen::Matrix3d> out_many_sols;
    start = chrono::steady_clock::now();
    for (int i = 0; i < nbr_iter; i++) {
        out_many_sols = EricssonResearch::solver_choi_kim_2018(data2);
    }
    end = chrono::steady_clock::now();
    cout << "Mean execution time (solver_choi_kim_2018 2pt): "
         << chrono::duration_cast<chrono::nanoseconds>(end - start).count() / nbr_iter << " ns" << endl;

    Vector2d Pi = Vector2d::Random();
    Vector2d Pj = Vector2d::Random();
    Matrix2d Ac = Matrix2d::Random();
    std::pair<VectorXd, VectorXd> out_sols2;

    // Guan et al. (CVPR 2020) - CS method
    start = chrono::steady_clock::now();
    for (int i = 0; i < nbr_iter; i++) {
        out_sols2 = EricssonResearch::solver_guan_cvpr_2020_cs(Pi, Pj, Ac);
    }
    end = chrono::steady_clock::now();
    cout << "Mean execution time (Guan et al. (CS)): "
         << chrono::duration_cast<chrono::nanoseconds>(end - start).count() / nbr_iter << " ns" << endl;
    return 0;
}
