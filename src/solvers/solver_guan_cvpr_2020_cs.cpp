#include <Eigen/Dense>
#include <cmath>

namespace EricssonResearch {
std::pair<Eigen::VectorXd, Eigen::VectorXd>
solver_guan_cvpr_2020_cs(const Eigen::Vector2d &Pi, const Eigen::Vector2d &Pj, const Eigen::Matrix2d &Ac) {
    const double u1 = Pi(0), v1 = Pi(1), u2 = Pj(0), v2 = Pj(1), a1 = Ac(0, 0), a2 = Ac(0, 1), a3 = Ac(1, 0),
                 a4 = Ac(1, 1);

    // Coefficient matrix
    Eigen::Matrix<double, 3, 4> C;
    C << v1, v1 * u2, v2, -u1 * v2, 0, a1 * v1, a3, -(a3 * u1 + v2), 1, (a2 * v1 + u2), a4, -a4 * u1;
    Eigen::EigenSolver<Eigen::Matrix4d> es(C.transpose() * C);
    Eigen::ArrayXXcd V = es.eigenvectors();

    // Sort out complex-valued
    const double thresh = 1e-5;
    Eigen::Array<double, 4, 1> real_sols;
    Eigen::VectorXd Yaw(4);
    Eigen::VectorXd AngleT(4);
    real_sols = V.imag().cwiseAbs().colwise().sum();
    int nroots = 0;
    Eigen::Array<double, 4, 1> tmp;

    for (int i = 0; i < 4; i++) {
        if (real_sols(i) <= thresh) {
            tmp = V.col(i).real();
            AngleT(nroots) = std::atan2(tmp(2), tmp(3)) / M_PI * 180;
            Yaw(nroots) = std::atan2(tmp(0), tmp(1)) / M_PI * 180 + AngleT(nroots);
            nroots++;
        }
    }
    Yaw.conservativeResize(nroots);
    AngleT.conservativeResize(nroots);
    return {Yaw, AngleT};
}
} // namespace EricssonResearch
