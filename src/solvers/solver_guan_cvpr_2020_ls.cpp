#include <Eigen/Dense>

using namespace Eigen;

namespace EricssonResearch {
Vector4d solver_guan_cvpr_2020_ls(const VectorXd &data) {
    // Compute coefficients
    const double *d = data.data();
    VectorXd coeffs(12);
    coeffs[0] = 1;
    coeffs[1] = std::pow(d[0], 2) + std::pow(d[4], 2) + std::pow(d[8], 2);
    coeffs[2] = d[0] * d[1] + d[4] * d[5] + d[8] * d[9];
    coeffs[3] = d[0] * d[2] + d[4] * d[6] + d[8] * d[10];
    coeffs[4] = d[0] * d[3] + d[4] * d[7] + d[8] * d[11];
    coeffs[5] = std::pow(d[1], 2) + std::pow(d[5], 2) + std::pow(d[9], 2);
    coeffs[6] = d[1] * d[2] + d[5] * d[6] + d[9] * d[10];
    coeffs[7] = d[1] * d[3] + d[5] * d[7] + d[9] * d[11];
    coeffs[8] = std::pow(d[2], 2) + std::pow(d[6], 2) + std::pow(d[10], 2);
    coeffs[9] = d[2] * d[3] + d[6] * d[7] + d[10] * d[11];
    coeffs[10] = std::pow(d[3], 2) + std::pow(d[7], 2) + std::pow(d[11], 2);
    coeffs[11] = -1;

    // Setup elimination template
    static const int coeffs0_ind[] = {
        0, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0, 0,
        0, 0, 1, 2, 4,  0, 2, 1, 2,  5, 4, 7,  2, 5, 7, 0, 1,  3, 2, 4,  0, 3,  2, 1,  2, 6, 3,  6, 9, 5,  4, 7, 3, 6,
        2, 5, 6, 9, 7,  0, 3, 8, 6,  9, 0, 3,  6, 8, 9, 0, 1,  3, 4, 0,  2, 4,  3, 9,  8, 3, 9,  4, 7, 10, 2, 6, 8, 9,
        6, 0, 4, 9, 10, 4, 2, 3, 7,  0, 9, 10, 9, 7, 6, 8, 10, 7, 9, 0,  0, 11, 0, 11, 4, 2, 1,  2, 6, 3,  7, 4, 7, 10,
        5, 0, 4, 7, 2,  5, 6, 7, 10, 0, 4, 7,  3, 6, 9, 8, 6,  9, 7, 10, 5, 0,  0, 4,  7, 9, 10, 7, 5, 6,  0, 0};
    static const int coeffs1_ind[] = {2, 4, 11, 11, 5, 7, 11, 11, 11, 11, 6, 9, 11, 11, 7, 10, 11, 11};
    static const int C0_ind[] = {
        0,    35,   43,   60,   86,   119,  129,  151,  172,  189,  215,  235,  258,  280,  301,  325,  344,
        364,  387,  419,  430,  459,  473,  485,  516,  526,  559,  583,  602,  629,  645,  669,  688,  702,
        731,  751,  756,  774,  776,  790,  798,  799,  800,  816,  817,  818,  841,  842,  859,  874,  885,
        891,  903,  905,  906,  924,  927,  928,  929,  933,  934,  942,  944,  945,  946,  947,  967,  968,
        970,  971,  976,  985,  988,  990,  1011, 1017, 1029, 1031, 1048, 1054, 1055, 1060, 1072, 1088, 1098,
        1103, 1106, 1118, 1123, 1137, 1140, 1143, 1145, 1147, 1148, 1150, 1155, 1157, 1161, 1165, 1189, 1192,
        1203, 1212, 1224, 1229, 1232, 1235, 1247, 1248, 1249, 1258, 1273, 1276, 1277, 1287, 1289, 1290, 1319,
        1331, 1332, 1338, 1376, 1379, 1419, 1423, 1428, 1434, 1435, 1436, 1439, 1440, 1442, 1443, 1446, 1448,
        1459, 1490, 1513, 1514, 1519, 1520, 1524, 1527, 1531, 1538, 1558, 1559, 1561, 1562, 1564, 1566, 1567,
        1569, 1570, 1576, 1581, 1623, 1657, 1687, 1688, 1692, 1695, 1697, 1709, 1710, 1718, 1751};
    static const int C1_ind[] = {32, 33, 40, 83, 116, 117, 118, 122, 165, 192, 200, 201, 235, 278, 284, 285, 288, 322};

    Matrix<double, 42, 42> C0;
    C0.setZero();
    Matrix<double, 42, 8> C1;
    C1.setZero();
    for (int i = 0; i < 168; i++) {
        C0(C0_ind[i]) = coeffs(coeffs0_ind[i]);
    }
    for (int i = 0; i < 18; i++) {
        C1(C1_ind[i]) = coeffs(coeffs1_ind[i]);
    }

    Matrix<double, 42, 8> C12 = C0.partialPivLu().solve(C1);

    // Setup action matrix
    Matrix<double, 16, 8> RR;
    RR << -C12.bottomRows(8), Matrix<double, 8, 8>::Identity(8, 8);

    static const int AM_ind[] = {0, 1, 2, 5, 3, 4, 6, 7};
    Matrix<double, 8, 8> AM;
    for (int i = 0; i < 8; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    Matrix<std::complex<double>, 6, 8> sols;
    sols.setZero();

    // Solve eigenvalue problem
    EigenSolver<Matrix<double, 8, 8>> es(AM);
    ArrayXcd D = es.eigenvalues();
    ArrayXXcd V = es.eigenvectors();
    ArrayXXcd scale = (D.transpose() / (V.row(2).array() * V.row(6).array())).sqrt();
    V = (V * scale.replicate(8, 1)).eval();

    sols.row(0) = V.row(0).array();
    sols.row(1) = V.row(2).array();
    sols.row(2) = V.row(4).array();
    sols.row(3) = V.row(6).array();
    sols.row(4) = V.row(5).array() / (sols.row(2).array());
    sols.row(5) = V.row(1).array() / (sols.row(0).array());

    // Pre-processing: Remove complex-valued solutions
    double thresh = 1e-5;
    Eigen::Array<double, 8, 1> real_sols;
    real_sols = sols.imag().cwiseAbs().colwise().sum();

    // Extract the true solution before returning
    const double a1 = d[0], b1 = d[1], c1 = d[2], d1 = d[3], a2 = d[4], b2 = d[5], c2 = d[6], d2 = d[7], a3 = d[8],
                 b3 = d[9], c3 = d[10], d3 = d[11];

    double minval = std::numeric_limits<double>::max();
    double val;
    Eigen::Vector4d best_sol;
    best_sol.setZero();
    Eigen::Vector4d x;

    for (int i = 0; i < 8; i++) {
        if (real_sols(i) <= thresh) {
            x = sols.col(i).head(4).real();

            val = std::pow(a1 * x[0] + b1 * x[1] + c1 * x[2] + d1 * x[3], 2) +
                  std::pow(a2 * x[0] + b2 * x[1] + c2 * x[2] + d2 * x[3], 2) +
                  std::pow(a3 * x[0] + b3 * x[1] + c3 * x[2] + d3 * x[3], 2);
            if (val < minval) {
                minval = val;
                best_sol = x;
            }
        }
    }
    return best_sol;
}
} // namespace EricssonResearch
// Action =  x2*x4
// Quotient ring basis (V) = x1,x1*x6,x2,x2*x6,x3,x3*x5,x4,x4*x5,
// Available monomials (RR*V) =
// x1*x2*x4,x1*x2*x4*x6,x2^2*x4,x2*x3*x4,x2*x3*x4*x5,x2^2*x4*x6,x2*x4^2,x2*x4^2*x5,x1,x1*x6,x2,x2*x6,x3,x3*x5,x4,x4*x5,
