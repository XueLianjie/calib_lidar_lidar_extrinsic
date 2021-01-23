
#include <iostream>
#include <vector>
#include <Eigen/Eigen>

#include <ceres/ceres.h>
#include "pose_local_parameterization.h"
#include <ceres/rotation.h>

Eigen::Matrix3d skew(const Eigen::Vector3d &p)
{
    Eigen::Matrix3d result;
    result << 0., -p.z(), p.y(),
        p.z(), 0., -p.x(),
        -p.y(), p.x(), 0.;
    return result;
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

Eigen::Vector2d homeTrans(Eigen::Vector3d &vec)
{
    Eigen::Vector2d result;
    result.x() = vec.x() / vec.z();
    result.y() = vec.y() / vec.z();
    return result;
}

struct PLData
{
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> points3d; // in lidar frame
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> lines2d;  // in image
};

void GenerateSimData(PLData &pl_data)
{
    Eigen::Matrix3d Rcl = Eigen::Matrix3d::Identity();
    Rcl << 0., -1., 0.,
        0., 0., -1.,
        1., 0., 0.;

    Eigen::Vector3d tcl(0.1, 0.01, 0.04);
    // Eigen::Vector3d tcl = Eigen::Vector3d::Zero();
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &points3d = pl_data.points3d; // lidar frame
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> &lines2d = pl_data.lines2d;   // cam frame

    double x = 50.;
    double y = 2.;
    double z = -2.;
    // first group
    Eigen::Vector3d c1 = Eigen::Vector3d(x, y, z);
    Eigen::Vector3d c2 = Eigen::Vector3d(x, y, -z);
    points3d.push_back(std::pair<Eigen::Vector3d, Eigen::Vector3d>(c1, c2));
    Eigen::Vector3d lc1 = Rcl * c1 + tcl;
    Eigen::Vector3d lc2 = Rcl * c2 + tcl;

    lines2d.push_back(std::pair<Eigen::Vector2d, Eigen::Vector2d>(homeTrans(lc1), homeTrans(lc2)));

    // second group
    Eigen::Vector3d r1 = Eigen::Vector3d(x, y, z);
    Eigen::Vector3d r2 = Eigen::Vector3d(x, -y, z);
    points3d.push_back(std::pair<Eigen::Vector3d, Eigen::Vector3d>(r1, r2));
    Eigen::Vector3d lr1 = Rcl * r1 + tcl;
    Eigen::Vector3d lr2 = Rcl * r2 + tcl;

    lines2d.push_back(std::pair<Eigen::Vector2d, Eigen::Vector2d>(homeTrans(lr1), homeTrans(lr2)));

    // first  group
    Eigen::Vector3d c11 = Eigen::Vector3d(x, -y, z);
    Eigen::Vector3d c12 = Eigen::Vector3d(x, -y, -z);
    points3d.push_back(std::pair<Eigen::Vector3d, Eigen::Vector3d>(c11, c12));
    Eigen::Vector3d lc11 = Rcl * c11 + tcl;
    Eigen::Vector3d lc12 = Rcl * c12 + tcl;

    lines2d.push_back(std::pair<Eigen::Vector2d, Eigen::Vector2d>(homeTrans(lc11), homeTrans(lc12)));

    // second group
    Eigen::Vector3d r11 = Eigen::Vector3d(x, y, -z);
    Eigen::Vector3d r12 = Eigen::Vector3d(x, -y, -z);
    points3d.push_back(std::pair<Eigen::Vector3d, Eigen::Vector3d>(r11, r12));
    Eigen::Vector3d lr11 = Rcl * r11 + tcl;
    Eigen::Vector3d lr12 = Rcl * r12 + tcl;

    lines2d.push_back(std::pair<Eigen::Vector2d, Eigen::Vector2d>(homeTrans(lr11), homeTrans(lr12)));

    return;
}

int main()
{
    Eigen::Matrix3d Rcl = Eigen::Matrix3d::Identity();
    Rcl << 0., -1., 0.,
        0., 0., -1.,
        1., 0., 0.;
    Eigen::Matrix3d R12;
    R12 = Eigen::AngleAxisd(M_PI / 18.0, Eigen::Vector3d::UnitY()); //初始旋转的扰动

    PLData pl_data;
    GenerateSimData(pl_data);
    Eigen::Quaterniond q( Rcl * R12);

    double pose[7] = {0.2, 0.2, 0.2, q.w(), q.x(), q.y(), q.z()};
    ceres::Problem problem;

    for (size_t i = 0; i < pl_data.points3d.size(); ++i)
    {

        Eigen::Vector3d &pl1 = pl_data.points3d[i].first;
        Eigen::Vector3d &pl2 = pl_data.points3d[i].second;

        Eigen::Vector2d &pc1 = pl_data.lines2d[i].first;
        Eigen::Vector2d &pc2 = pl_data.lines2d[i].second;

        // Eigen::Vector2d& pc3 = pl_data.lines2d.back().first;
        // Eigen::Vector2d& pc4 = pl_data.lines2d.back().second;

        Eigen::Vector3d l1(pc2.y() - pc1.y(), pc1.x() - pc2.x(), pc2.x() * pc1.y() - pc2.y() * pc1.x());
        // Eigen::Vector3d l2(pc4.y() - pc3.y(), pc3.x() - pc4.x(), pc4.x() * pc3.y() - pc4.y() * pc3.x());

        ceres::CostFunction *cost_function1 = new ceres::AutoDiffCostFunction<PLDistanceFunctor, 1, 4, 3>(new PLDistanceFunctor(pl1, l1));
        problem.AddResidualBlock(cost_function1, NULL, pose + 3, pose);

        ceres::CostFunction *cost_function2 = new ceres::AutoDiffCostFunction<PLDistanceFunctor, 1, 4, 3>(new PLDistanceFunctor(pl2, l1));
        problem.AddResidualBlock(cost_function2, NULL, pose + 3, pose);
    }

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

    return 0;
}
