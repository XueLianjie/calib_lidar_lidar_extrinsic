#include "ros/ros.h"
#include <vector>
#include <Eigen/Eigen>
#include <iostream>
#include <ceres/ceres.h>
#include "pose_local_parameterization.h"
#include <ceres/rotation.h>
#define USE_AUTODIFF
Eigen::Matrix3d skew(const Eigen::Vector3d &p)
{
    Eigen::Matrix3d result;
    result << 0., -p.z(), p.y(),
        p.z(), 0., -p.x(),
        -p.y(), p.x(), 0.;
    return result;
}

struct PLData
{
    std::vector<Eigen::Vector3d> points3d;
    std::vector<Eigen::Vector3d> lines2d;
};

void GenerateLidarCamSimData(PLData &pl_data)
{
    Eigen::Matrix3d Rcl = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d R12;
    R12 = Eigen::AngleAxisd(M_PI / 10.0, Eigen::Vector3d::UnitY());
    Eigen::Vector3d tcl(0., 0., 0.);

    std::vector<Eigen::Vector3d> &points3d = pl_data.points3d;
    std::vector<Eigen::Vector3d> &lines2d = pl_data.lines2d;

    double x = 1.;
    double y = -0.5;
    double z = -0.5;
    for (size_t i = 1; i < 10; ++i)
    {
        y += 0.1;
        std::cout << "x y z " << x << " " << y << " " << z << std::endl;
        points3d.push_back(Rcl * Eigen::Vector3d(x, y, z) + tcl);
    }

    y = -0.5;
    z = -0.5;
    for (size_t i = 1; i < 10; ++i)
    {
        z += 0.1;
        std::cout << "x y z " << x << " " << y << " " << z << std::endl;
        points3d.push_back(Rcl * Eigen::Vector3d(x, y, z) + tcl);
    }

    y = -0.5;
    z = 0.5;
    for (size_t i = 1; i < 10; ++i)
    {
        y += 0.1;
        std::cout << "x y z " << x << " " << y << " " << z << std::endl;
        points3d.push_back(Rcl * Eigen::Vector3d(x, y, z) + tcl);
    }

    y = 0.5;
    z = -0.5;
    for (size_t i = 1; i < 10; ++i)
    {
        z += 0.1;
        std::cout << "x y z " << x << " " << y << " " << z << std::endl;
        points3d.push_back(Rcl * Eigen::Vector3d(x, y, z) + tcl);
    }
    std::cout << "points3d size " << points3d.size() << std::endl;

    tcl += Eigen::Vector3d(0.2, 0.3, 0.1);
    Rcl << 0., -1., 0.,
        0., 0., -1.,
        1., 0., 0.;
    Eigen::Vector3d pl2(x, -0.5, -0.5), pl3(x, -0.5, 0.5), pl4(x, 0.5, 0.5), pl1(x, 0.5, -0.5);
    Eigen::Vector3d tmp = Rcl * pl1 + tcl;
    Eigen::Vector3d pc1(tmp.x() / tmp.z(), tmp.y() / tmp.z(), 1.);
    tmp = Rcl * pl2 + tcl;
    Eigen::Vector3d pc2(tmp.x() / tmp.z(), tmp.y() / tmp.z(), 1.);
    tmp = Rcl * pl3 + tcl;
    Eigen::Vector3d pc3(tmp.x() / tmp.z(), tmp.y() / tmp.z(), 1.);
    tmp = Rcl * pl4 + tcl;
    Eigen::Vector3d pc4(tmp.x() / tmp.z(), tmp.y() / tmp.z(), 1.);
    std::cout << "pc 1 2 3 4 \n"
              << pc1.transpose() << " " << pc2.transpose() << " " << pc3.transpose() << " " << pc4.transpose() << std::endl;

    Eigen::Vector3d l1(pc2.y() - pc1.y(), pc1.x() - pc2.x(), pc2.x() * pc1.y() - pc2.y() * pc1.x());
    Eigen::Vector3d l2(pc3.y() - pc2.y(), pc2.x() - pc3.x(), pc3.x() * pc2.y() - pc3.y() * pc2.x());
    Eigen::Vector3d l3(pc4.y() - pc3.y(), pc3.x() - pc4.x(), pc4.x() * pc3.y() - pc4.y() * pc3.x());
    Eigen::Vector3d l4(pc1.y() - pc4.y(), pc4.x() - pc1.x(), pc1.x() * pc4.y() - pc1.y() * pc4.x());
    std::cout << "l 1 2 3 4 \n"
              << l1.transpose() << " " << l2.transpose() << " " << l3.transpose() << " " << l4.transpose() << std::endl;

    lines2d.push_back(l1);
    lines2d.push_back(l2);
    lines2d.push_back(l3);
    lines2d.push_back(l4);
}

struct PLDistanceFunctor
{
    PLDistanceFunctor(const Eigen::Vector3d &p, const Eigen::Vector3d &l) : pl(p), lc(l)
    {
    }
    template <typename T>
    bool operator()(const T *const rotation, const T *const trans, T *residual) const
    {
        T pc[3];
        T p2[3] = {T(pl.x()), T(pl.y()), T(pl.z())};
        T l[3] = {T(lc.x()), T(lc.y()), T(lc.z())};
        T D = sqrt(l[0] * l[0] + l[1] * l[1]);
        ceres::QuaternionRotatePoint(rotation, p2, pc); // 用ceres 自带的QuaternionRotatePoint可以使用自动求导
        pc[0] += trans[0];
        pc[1] += trans[1];
        pc[2] += trans[2];

        residual[0] = (pc[0] / pc[2] * l[0] + pc[1] / pc[2] * l[1] + l[2]) / D; //  - T(point1_.x()) );

        return true;
    }

private:
    Eigen::Vector3d pl;
    Eigen::Vector3d lc;
};

class PLDistanceFactor : public ceres::SizedCostFunction<1, 7>
{
public:
    PLDistanceFactor(const Eigen::Vector3d &p, const Eigen::Vector3d &l) : pl(p), lc(l)
    {
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d tcl(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond qcl(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d pc = qcl.toRotationMatrix() * pl + tcl;
        double weight = sqrt(lc.x() * lc.x() + lc.y() * lc.y());
        double distance = ((lc.x() * pc.x() / pc.z() + lc.y() * pc.y() / pc.z() + lc.z())) / weight;
        residuals[0] = distance;
        //std::cout << "weight " << weight << "residual " << residuals[0] << std::endl;

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Matrix<double, 1, 3> dr_dp;
                dr_dp << lc.x() / (weight * pc.z()), lc.y() / (weight * pc.z()), -(lc.x() * pc.x() + lc.y() * pc.y()) / (pc.z() * pc.z() * weight); //这里容易出错，一定要小心
                Eigen::Matrix<double, 3, 6> dp_dpose;
                dp_dpose.leftCols<3>() = Eigen::Matrix3d::Identity();
                dp_dpose.rightCols<3>() = -qcl.toRotationMatrix() * skew(pl);
                Eigen::Matrix<double, 1, 6> jacob_i = dr_dp * dp_dpose;

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacob(jacobians[0]);
                jacob.leftCols<6>() = jacob_i;
                jacob.rightCols<1>().setZero();
            }
        }

        return true;
    }

private:
    Eigen::Vector3d pl;
    Eigen::Vector3d lc;
};

int main(int argc, char **argv)
{
    PLData pl_data;
    GenerateLidarCamSimData(pl_data);
#ifdef USE_AUTODIFF
    ceres::Problem problem;

        Eigen::Matrix3d Rcl = Eigen::Matrix3d::Identity();
    Rcl << 0., -1., 0.,
        0., 0., -1.,
        1., 0., 0.;
    Eigen::Matrix3d R12;
    R12 = Eigen::AngleAxisd(M_PI / 10.0, Eigen::Vector3d::UnitY()); //初始旋转的扰动
    std::cout << "R12 " << R12 << std::endl;

    Eigen::Quaterniond q(  Rcl  ); //旋转矩阵的扰动左转和右转好像不太一样

    double pose[7] = {0., 3., 0., q.w(), q.x(), q.y(), q.z() };

    for (size_t i = 0; i < 9; ++i)
    {
        ceres::CostFunction *pl_factor = new ceres::AutoDiffCostFunction<PLDistanceFunctor, 1, 4, 3>(new PLDistanceFunctor(pl_data.points3d[i], pl_data.lines2d[0]));
        problem.AddResidualBlock(pl_factor, NULL, pose + 3, pose);
    }

    for (size_t i = 9; i < 18; ++i)
    {
        ceres::CostFunction *pl_factor = new ceres::AutoDiffCostFunction<PLDistanceFunctor, 1, 4, 3>(new PLDistanceFunctor(pl_data.points3d[i], pl_data.lines2d[1]));
        problem.AddResidualBlock(pl_factor, NULL, pose + 3, pose);
    }

    for (size_t i = 18; i < 27; ++i)
    {
        ceres::CostFunction *pl_factor = new ceres::AutoDiffCostFunction<PLDistanceFunctor, 1, 4, 3>(new PLDistanceFunctor(pl_data.points3d[i], pl_data.lines2d[2]));
        problem.AddResidualBlock(pl_factor, NULL, pose + 3, pose);
    }

    for (size_t i = 27; i < 36; ++i)
    {
        ceres::CostFunction *pl_factor = new ceres::AutoDiffCostFunction<PLDistanceFunctor, 1, 4, 3>(new PLDistanceFunctor(pl_data.points3d[i], pl_data.lines2d[3]));
        problem.AddResidualBlock(pl_factor, NULL, pose + 3, pose);
    }

    // ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    // problem.AddParameterBlock(pose, 7, local_parameterization);
    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    Eigen::Quaterniond quat = Eigen::Quaterniond(pose[3], pose[4], pose[5], pose[6]);
    // Eigen::Map<Eigen::Quaterniond> qq(pose + 3);
    // std::cout << "qq " << qq.toRotationMatrix() << std::endl;
    Eigen::Vector3d translation(pose);
    std::cout << "q " << quat.toRotationMatrix() << std::endl;
    std::cout << "t " << translation << std::endl;
#else
    ceres::Problem problem;
    Eigen::Matrix3d Rcl = Eigen::Matrix3d::Identity();
    Rcl << 0., -1., 0.,
        0., 0., -1.,
        1., 0., 0.;
    Eigen::Matrix3d R12;
    R12 = Eigen::AngleAxisd(M_PI / 3.0, Eigen::Vector3d::UnitY()); //初始旋转的扰动
    Eigen::Quaterniond q( R12  * Rcl  );

    double pose[7] = {0., 0., 0., q.x(), q.y(), q.z(), q.w()};
    for (size_t i = 0; i < 2; ++i)
    {
        PLDistanceFactor *pl_factor = new PLDistanceFactor(pl_data.points3d[i], pl_data.lines2d[0]);
        problem.AddResidualBlock(pl_factor, NULL, pose);
    }

    for (size_t i = 9; i < 11; ++i)
    {
        PLDistanceFactor *pl_factor = new PLDistanceFactor(pl_data.points3d[i], pl_data.lines2d[1]);
        problem.AddResidualBlock(pl_factor, NULL, pose);
    }

    for (size_t i = 18; i < 20; ++i)
    {
        PLDistanceFactor *pl_factor = new PLDistanceFactor(pl_data.points3d[i], pl_data.lines2d[2]);
        problem.AddResidualBlock(pl_factor, NULL, pose);
    }

    for (size_t i = 27; i < 29; ++i)
    {
        PLDistanceFactor *pl_factor = new PLDistanceFactor(pl_data.points3d[i], pl_data.lines2d[3]);
        problem.AddResidualBlock(pl_factor, NULL, pose);
    }

    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(pose, 7, local_parameterization);
    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    Eigen::Quaterniond quat = Eigen::Quaterniond(pose[6], pose[3], pose[4], pose[5]);
    // Eigen::Map<Eigen::Quaterniond> qq(pose + 3);
    // std::cout << "qq " << qq.toRotationMatrix() << std::endl;
    Eigen::Vector3d translation(pose);
    std::cout << "qq " << quat.toRotationMatrix() << std::endl;
    std::cout << "tt " << translation << std::endl;
#endif
    return 0;
}