
#include <iostream>
#include <vector>
#include <Eigen/Eigen>

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

struct PLDistanceFunctor
{
    PLDistanceFunctor(const Eigen::Vector3d &p, const Eigen::Vector3d &l) : pl(p), lc(l)
    {
    }
    template <typename T>
    bool operator()(const T *const rotation, const T *const trans, T *residual) const
    {
        Eigen::Quaternion<T> quat{rotation[3], rotation[0], rotation[1], rotation[2]}; //如果这么用，相当于用了Eigen里面的四元数的内部运算规则了，也需要进行EigenQuaternionParameterization了
        Eigen::Matrix<T, 3, 1> t{trans[0], trans[1], trans[2]};
        Eigen::Matrix<T, 3, 1> p_l{T(pl.x()), T(pl.y()), T(pl.z())};
        Eigen::Matrix<T, 3, 1> l_c{T(lc.x()), T(lc.y()), T(lc.z())};
        Eigen::Matrix<T, 3, 1> p_c;
        p_c = quat * p_l + t;
        residual[0] = ((p_c[0] / p_c[2] * l_c[0] + p_c[1] / p_c[2] * l_c[1] + l_c[2]) / sqrt(l_c[0] * l_c[0] + l_c[1] * l_c[1]));
        return true;
        // T pc[3];
        // T p2[3] = {T(pl.x()), T(pl.y()), T(pl.z())};
        // T l[3] = {T(lc.x()), T(lc.y()), T(lc.z())};
        // T D = sqrt(l[0] * l[0] + l[1] * l[1]);
        // ceres::QuaternionRotatePoint(rotation, p2, pc); // 用ceres 自带的QuaternionRotatePoint可以使用自动求导，这样用的话之后需要addparameterization ceres::Quaternionparameterization
        // pc[0] += trans[0];
        // pc[1] += trans[1];
        // pc[2] += trans[2];

        // residual[0] = ((pc[0] / pc[2] * l[0] + pc[1] / pc[2] * l[1] + l[2]) / D); //  - T(point1_.x()) );

        // return true;
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
    std::cout << "result " << result.x() << " " << result.y() << std::endl;
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

    double x = 43.;
    double y = -1.5;
    double z = -1.5;
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

class RotationFactor : public ceres::SizedCostFunction<1, 4>
{
public:
    RotationFactor(const Eigen::Vector3d &p, const Eigen::Vector3d &l) : pl(p), lc(l)
    {
    }
    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond qcl(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]);
        Eigen::Vector3d pc = qcl.toRotationMatrix() * pl;
        double weight = sqrt(lc.x() * lc.x() + lc.y() * lc.y());
        double distance = ((lc.x() * pc.x() / pc.z() + lc.y() * pc.y() / pc.z() + lc.z())) / weight;
        residuals[0] = distance;

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Matrix<double, 1, 3> dr_dp;
                dr_dp << lc.x() / (weight * pc.z()), lc.y() / (weight * pc.z()), -(lc.x() * pc.x() + lc.y() * pc.y()) / (pc.z() * pc.z() * weight); //这里容易出错，一定要小心
                Eigen::Matrix<double, 3, 3> dp_dpose;
                dp_dpose = -qcl.toRotationMatrix() * skew(pl);
                Eigen::Matrix<double, 1, 3> jacob_i = dr_dp * dp_dpose;

                Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> jacob(jacobians[0]);
                jacob.leftCols<3>() = jacob_i;
                jacob.rightCols<1>().setZero();
            }
        }

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

int main()
{
    Eigen::Matrix3d Rcl = Eigen::Matrix3d::Identity();
    Rcl << 0., -1., 0.,
        0., 0., -1.,
        1., 0., 0.;
    Eigen::Matrix3d R12;
    R12 = Eigen::AngleAxisd(-M_PI / 6.0, Eigen::Vector3d::UnitY()); //初始旋转的扰动
    std::cout << "R12 " << R12 << std::endl;

    PLData pl_data;
    GenerateSimData(pl_data);
    Eigen::Quaterniond q( R12 * Rcl );

#ifndef USE_AUTODIFF
    double pose[7] = {0.1, 0.01, 0.04, q.x(), q.y(), q.z(), q.w()};
    ceres::Problem problem;

    for (size_t i = 0; i < pl_data.points3d.size(); ++i)
    {

        Eigen::Vector3d &pl1 = pl_data.points3d[i].first;
        Eigen::Vector3d &pl2 = pl_data.points3d[i].second;
        std::cout << "pl1 " << pl1.transpose() << std::endl;
        std::cout << "pl2 " << pl2.transpose() << std::endl;

        Eigen::Vector2d &pc1 = pl_data.lines2d[i].first;
        Eigen::Vector2d &pc2 = pl_data.lines2d[i].second;
        std::cout << "pc1 " << pc1.transpose() << std::endl;
        std::cout << "pc2 " << pc2.transpose() << std::endl;

        // Eigen::Vector2d& pc3 = pl_data.lines2d.back().first;
        // Eigen::Vector2d& pc4 = pl_data.lines2d.back().second;

        Eigen::Vector3d l1(pc2.y() - pc1.y(), pc1.x() - pc2.x(), pc2.x() * pc1.y() - pc2.y() * pc1.x());
        // Eigen::Vector3d l2(pc4.y() - pc3.y(), pc3.x() - pc4.x(), pc4.x() * pc3.y() - pc4.y() * pc3.x());
        std::cout << "l1 " << l1.transpose() << std::endl;

        PLDistanceFactor *pl_factor1 = new PLDistanceFactor(pl1, l1);
        problem.AddResidualBlock(pl_factor1, NULL, pose);
        PLDistanceFactor *pl_factor = new PLDistanceFactor(pl2, l1);
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
    std::cout << "q " << quat.toRotationMatrix() << std::endl;
    std::cout << "t " << translation << std::endl;
#else

    double pose[7] = {0.3, 0.41, 0.04, q.x(), q.y(), q.z(), q.w()};
    ceres::Problem problem;

    for (size_t i = 0; i < pl_data.points3d.size(); ++i)
    {

        Eigen::Vector3d &pl1 = pl_data.points3d[i].first;
        Eigen::Vector3d &pl2 = pl_data.points3d[i].second;
        std::cout << "pl1 " << pl1.transpose() << std::endl;
        std::cout << "pl2 " << pl2.transpose() << std::endl;

        Eigen::Vector2d &pc1 = pl_data.lines2d[i].first;
        Eigen::Vector2d &pc2 = pl_data.lines2d[i].second;
        std::cout << "pc1 " << pc1.transpose() << std::endl;
        std::cout << "pc2 " << pc2.transpose() << std::endl;

        // Eigen::Vector2d& pc3 = pl_data.lines2d.back().first;
        // Eigen::Vector2d& pc4 = pl_data.lines2d.back().second;

        Eigen::Vector3d l1(pc2.y() - pc1.y(), pc1.x() - pc2.x(), pc2.x() * pc1.y() - pc2.y() * pc1.x());
        // Eigen::Vector3d l2(pc4.y() - pc3.y(), pc3.x() - pc4.x(), pc4.x() * pc3.y() - pc4.y() * pc3.x());
        std::cout << "l1 " << l1.transpose() << std::endl;

        ceres::CostFunction *cost_function1 = new ceres::AutoDiffCostFunction<PLDistanceFunctor, 1, 4, 3>(new PLDistanceFunctor(pl1, l1));
        problem.AddResidualBlock(cost_function1, NULL, pose + 3, pose);

        ceres::CostFunction *cost_function2 = new ceres::AutoDiffCostFunction<PLDistanceFunctor, 1, 4, 3>(new PLDistanceFunctor(pl2, l1));
        problem.AddResidualBlock(cost_function2, NULL, pose + 3, pose);
    }
    ceres::LocalParameterization *q_parameterization =
        new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(pose+3, 4, q_parameterization);
    problem.AddParameterBlock(pose, 3);
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
    std::cout << "q " << quat.toRotationMatrix() << std::endl;
    std::cout << "t " << translation << std::endl;
#endif

    return 0;
}
