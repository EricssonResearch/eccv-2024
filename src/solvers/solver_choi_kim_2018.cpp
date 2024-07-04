#include <Eigen/Dense>
#include <algorithm>
#include <vector>

namespace EricssonResearch {
std::vector<Eigen::Matrix3d> solver_choi_kim_2018(const Eigen::Matrix<double, 8, 1> &data_) {
    const double *data_ptr = data_.data();
    const double u11 = data_ptr[0], v11 = data_ptr[1], u21 = data_ptr[2], v21 = data_ptr[3], u12 = data_ptr[4],
                 v12 = data_ptr[5], u22 = data_ptr[6], v22 = data_ptr[7];

    // Building the coefficient matrices
    Eigen::Matrix2d A1, A2;

    A1(0, 0) = v11;
    A1(0, 1) = -v11 * u21;
    A2(0, 0) = -v21;
    A2(0, 1) = u11 * v21;
    A1(1, 0) = v12;
    A1(1, 1) = -v12 * u22;
    A2(1, 0) = -v22;
    A2(1, 1) = u12 * v22;

    Eigen::Matrix2d pseudoInverseA2 = (A2.transpose() * A2).inverse() * A2.transpose();
    Eigen::Matrix2d C = -pseudoInverseA2 * A1;
    Eigen::Matrix2d C2 = C.transpose() * C;

    double a = C2(0, 0);
    double b = C2(1, 0);
    double c = C2(1, 1);

    double CF1 = 4 * b * b + (a - c) * (a - c);
    double CF2 = -2 * (a - c) * (2 - a - c);
    double CF3 = (2 - a - c) * (2 - a - c) - 4 * b * b;

    double cos2alpha1 = (-CF2 + sqrt(CF2 * CF2 - 4 * CF1 * CF3)) / (2 * CF1);
    double cos2alpha2 = (-CF2 - sqrt(CF2 * CF2 - 4 * CF1 * CF3)) / (2 * CF1);

    std::vector<double> alphas = {acos(cos2alpha1) / 2, acos(cos2alpha2) / 2, -acos(cos2alpha1) / 2,
                                  -acos(cos2alpha2) / 2};

    double alpha;
    Eigen::Vector2d v1, v2;
    std::vector<Eigen::Matrix3d> output;

    for (size_t pose_idx = 0; pose_idx < 4; ++pose_idx) {
        alpha = alphas[pose_idx];

        v1(0) = cos(alpha);
        v1(1) = sin(alpha);

        v2 = C * v1;
        v2 = v2 / v2.norm();

        double alpha = atan2(v1(1), v1(0));
        double alphaminusbeta = atan2(v2(1), v2(0));
        // double beta = alpha - alphaminusbeta;

        Eigen::Matrix3d model;
        model << 0, -sin(alpha), 0, sin(alphaminusbeta), 0, -cos(alphaminusbeta), 0, cos(alpha), 0;
        output.push_back(model);
    }
    return output;
}
} // namespace EricssonResearch
