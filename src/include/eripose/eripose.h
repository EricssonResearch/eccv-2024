#ifndef ERIPOSE_H_
#define ERIPOSE_H_

#include <Eigen/Dense>
#include <vector>

namespace EricssonResearch {
// Polynomial solvers
Eigen::Vector4d solver_guan_cvpr_2020_ls(const Eigen::VectorXd &data);
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> solver_valtonen_ornhag_eccv_2024(const Eigen::Matrix<double, 8, 1> &data_);
std::pair<Eigen::VectorXd, Eigen::VectorXd>
solver_guan_cvpr_2020_cs(const Eigen::Vector2d &Pi, const Eigen::Vector2d &Pj, const Eigen::Matrix2d &Ac);
std::vector<Eigen::Matrix3d> solver_choi_kim_2018(const Eigen::Matrix<double, 8, 1> &data_);
} // namespace EricssonResearch
#endif // ERIPOSE_H_
