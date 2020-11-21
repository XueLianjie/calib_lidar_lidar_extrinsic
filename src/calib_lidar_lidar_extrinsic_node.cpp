#include <iostream>

#include <ros/ros.h>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <random>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

struct ICPCostFunctor
{
    ICPCostFunctor(Eigen::Vector3d &point1, Eigen::Vector3d &point2) : point1_(point1), point2_(point2)
    {
    }
    // parameters q x y z w,Eigen中存储的顺序是x y z w,而初始化的顺序是w x y z
    template <typename T>
    bool operator()(const T *const rotation, const T* const t, T *residual) const // 返回类型是bool值
    {
        T p[3];
        T p2[3] = {T(point2_.x()), T(point2_.y()), T(point2_.z())};

        ceres::QuaternionRotatePoint(rotation, p2, p);

        //  Eigen::Map<const Eigen::Quaterniond> q(parameters);
        // Eigen::Vector3d t(parameters + 4);
        //Eigen::Vector3d res = q.toRotationMatrix() * point2_ + t - point1_;
        residual[0] = p[0] + t[0] - T(point1_.x());
        residual[1] = p[1] + t[1] - T(point1_.y());
        residual[2] = p[2] + t[2] - T(point1_.z());
        return true;
    }

private:
    Eigen::Vector3d point1_;
    Eigen::Vector3d point2_;
};

bool GenerateSimData(vector<Eigen::Vector3d> &points1, vector<Eigen::Vector3d> &points2)
{
    std::random_device random_d;
    std::default_random_engine gererator(random_d());
    std::uniform_real_distribution<double> z_rand(7, 10);
    std::uniform_real_distribution<double> rpy_rand(-M_PI / 6., M_PI / 6.);

    Eigen::Matrix3d R12;
    R12 = Eigen::AngleAxisd(M_PI / 6.0, Eigen::Vector3d::UnitY());

    // R12 << 0, 0, 1,
    //     -1, 0, 0,
    //     0, -1, 0;

    Eigen::Quaterniond q12(R12);
    Eigen::Vector3d t12(0., 0., 0.3);
    cout << "===================== Ground Truth ======================" << endl;
    cout << R12 << endl;
    cout << t12 << endl;
    cout << "===================== End ===============================" << endl;

    for (size_t i = 0; i < 7; ++i)
    {
        Eigen::Vector3d point(-3.0 + i, -0.2, 10.0);
        points1.push_back(point);
        point.y() = -point.y();
        points1.push_back(point);
        cout << "point2 " << point << endl;
    }

    for (auto &p : points1)
    {
        points2.push_back(R12 * p + t12);
        cout << "point2 " << R12 * p + t12 << endl;
    }
    cout << "points1 size " << points1.size() << endl;
    cout << "points2 size " << points2.size() << endl;

    return true;
}

Eigen::Matrix4d CaculateTransformation(vector<Eigen::Vector3d> points1, vector<Eigen::Vector3d> points2)
{

    Eigen::Vector3d m1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d m2 = Eigen::Vector3d::Zero();
    int size = points1.size();
    for (auto &p : points1)
    {
        m1 += p;
    }
    for (auto &p : points2)
    {
        m2 += p;
    }
    m1 /= size;
    m2 /= size;
    cout << "m1 " << m1 << endl;
    cout << "m2 " << m2 << endl;

    vector<Eigen::Vector3d> q1(size), q2(size);
    for (size_t i = 0; i < size; ++i)
    {
        q1[i] = points1[i] - m1;
        q2[i] = points2[i] - m2;
    }

    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < size; ++i)
    {
        W += Eigen::Vector3d(q1[i].x(), q1[i].y(), q1[i].z()) * (Eigen::Vector3d(q2[i].x(), q2[i].y(), q2[i].z()).transpose());
    }
    cout << "W = " << W << endl;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout << "U = " << U << endl;
    cout << "V = " << V << endl;

    Eigen::Matrix3d R = U * (V.transpose());
    Eigen::Vector3d t = Eigen::Vector3d(m1.x(), m1.y(), m1.z()) - R * Eigen::Vector3d(m2.x(), m2.y(), m2.z());

    cout << "R " << R << endl;
    cout << "t " << t << endl;

    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();

    transformation.block<3, 3>(0, 0) = R;
    transformation.block<3, 1>(0, 3) = t;

    return transformation;
}

int main(int argc, char **argv)
{
    cout << "start generate simulation data." << endl;
    vector<Eigen::Vector3d> points1, points2;

    double q[7] = {1., 0., 0., 0., 0., 0., 0.};

    ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	problem.AddParameterBlock(q, 4, local_parameterization);
	problem.AddParameterBlock(q + 4, 3);

    if (GenerateSimData(points1, points2))
    {
        Eigen::Matrix4d trans = CaculateTransformation(points2, points1);
        cout << "transformation result " << trans << endl;
        for (size_t i = 0; i < points1.size(); ++i)
        {
            ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<ICPCostFunctor, 3, 4, 3>(new ICPCostFunctor(points2[i], points1[i]));
            problem.AddResidualBlock(cost_function, NULL, q, q + 4);
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        //options.minimizer_progress_to_stdout = true;
        options.max_solver_time_in_seconds = 0.2;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
        Eigen::Quaterniond quat = Eigen::Quaterniond(q[0], q[1], q[2], q[3]);
        Eigen::Map<Eigen::Quaterniond> qq(q);
        cout << "qq " << qq.toRotationMatrix() << endl;
        Eigen::Vector3d translation(q + 4);
        cout << "q " << quat.toRotationMatrix() << endl;
        cout << "t " << translation << endl;

    }
    else
    {
        cout << "generate sim data error! " << endl;
    }

    return 0;
}