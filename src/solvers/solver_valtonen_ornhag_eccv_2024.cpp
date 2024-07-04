#include <Eigen/Dense>
#include <algorithm>

namespace EricssonResearch {
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> solver_valtonen_ornhag_eccv_2024(const Eigen::Matrix<double, 8, 1> &data_) {
    const double *data_ptr = data_.data();
    const double u1 = data_ptr[0], v1 = data_ptr[1], u2 = data_ptr[2], v2 = data_ptr[3], q1 = data_ptr[4],
                 q2 = data_ptr[5], o1 = data_ptr[6], o2 = data_ptr[7];

    const double s1 = sin(o1), c1 = cos(o1), s2 = sin(o2), c2 = cos(o2), q = q2 / q1;

    // Coefficient matrix
    double a = v1 * u2;
    double b = u1 * v2;
    double c = v2;
    double d = v1;
    double e = v1 * q * c2 + u2 * s1;
    double f = u1 * q * s2 + v2 * c1;
    double g = q * s2;
    double h = s1;

    Eigen::Matrix<double, 2, 4> M1;

    M1 << a, b, c, d, e, f, g, h;
    Eigen::JacobiSVD<Eigen::Matrix<double, 2, 4>> svd(M1, Eigen::ComputeFullV);
    Eigen::MatrixXd N = svd.matrixV().rightCols(2);
    double n1 = N(0, 0), n2 = N(1, 0), n3 = N(2, 0), n4 = N(3, 0), m1 = N(0, 1), m2 = N(1, 1), m3 = N(2, 1),
           m4 = N(3, 1);

    // Coefficients for quadratic equation
    double k2 = n1 * n1 - n2 * n2 - n3 * n3 + n4 * n4;
    double k1 = 2 * m1 * n1 - 2 * m2 * n2 - 2 * m3 * n3 + 2 * m4 * n4;
    double k0 = m1 * m1 - m2 * m2 - m3 * m3 + m4 * m4;

    Eigen::Matrix3d E1, E2;
    E1.setIdentity();
    E2.setIdentity();

    // Find (positive) roots
    double disc = k1 * k1 - 4 * k2 * k0;
    if (disc < 0) {
        return {E1, E2};
    }
    double sdisc = sqrt(disc);
    double r1 = (-k1 + sdisc) / (2 * k2);
    double r2 = (-k1 - sdisc) / (2 * k2);

    // Reconstruct essential matrices (Two possible solutions)
    E1 << 0, n1 * r1 + m1, 0, n2 * r1 + m2, 0, n3 * r1 + m3, 0, n4 * r1 + m4, 0;
    E2 << 0, n1 * r2 + m1, 0, n2 * r2 + m2, 0, n3 * r2 + m3, 0, n4 * r2 + m4, 0;

    return {E1, E2};
}
} // namespace EricssonResearch
