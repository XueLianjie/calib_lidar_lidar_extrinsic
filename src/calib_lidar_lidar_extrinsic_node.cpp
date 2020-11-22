#include <iostream>

#include <ros/ros.h>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <random>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "pose_local_parameterization.h"

using namespace std;
    Eigen::Matrix3d skew(const Eigen::Vector3d& p)
    {
        Eigen::Matrix3d result;
        result << 0., -p.z(), p.y(),
                  p.z(), 0.,  -p.x(),
                  -p.y(), p.x(), 0.;
        return result;
    }
class PnPFactor : public ceres::SizedCostFunction<2, 7>
{
public:
    PnPFactor(const Eigen::Vector3d& point3d, const Eigen::Vector2d& point2d):point3d_(point3d), point2d_(point2d)
    {}



    bool Evaluate( double const * const * parameters, double* residuals, double** jacobians) const 
    {
        Eigen::Vector3d t21(parameters[0]);
        Eigen::Quaterniond q21(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Vector3d p21 = q21.toRotationMatrix() * point3d_ + t21;
        residuals[0] = p21.x() / p21.z() - point2d_.x();
        residuals[1] = p21.y() / p21.z() - point2d_.y();

        if(jacobians)
        {
            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                Eigen::Matrix<double, 2, 6> jacob_i; // Eigen中默认的是ColMajor存储顺序
                Eigen::Matrix<double, 2, 3> dr_dp;
                dr_dp << 1./p21.z(), 0., -p21.x()/(p21.z() * p21.z()),
                         0., 1./p21.z(), -p21.y()/(p21.z() * p21.z());

                Eigen::Matrix<double, 3, 6> dp_dpose;
                dp_dpose.leftCols<3>().setIdentity();
                dp_dpose.rightCols<3>() = -q21.toRotationMatrix() * skew(point3d_);
                jacob_i = dr_dp * dp_dpose;
                jacobian_pose_i.leftCols<6>() = jacob_i;
                jacobian_pose_i.rightCols<1>().setZero();
            }


        }

        return true;
    }

private:
    Eigen::Vector3d point3d_;
    Eigen::Vector2d point2d_;

};



// 自动求导仿函数，需要采用模板方法重载（）操作符，内部的矩阵运算也不好直接采用Eigen的库，而是采用ceres自带的矩阵操作的模板方法.
struct ICPCostFunctor
{
    ICPCostFunctor(Eigen::Vector3d &point1, Eigen::Vector3d &point2) : point1_(point1), point2_(point2)
    {
    }
    // parameters q x y z w,Eigen中存储的顺序是x y z w,而初始化的顺序是w x y z
    template <typename T>
    bool operator()(const T *const rotation, const T *const t, T *residual) const // 返回类型是bool值
    {
        T p[3];
        T p2[3] = {T(point2_.x()), T(point2_.y()), T(point2_.z())};

        ceres::QuaternionRotatePoint(rotation, p2, p); // 用ceres 自带的QuaternionRotatePoint可以使用自动求导

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

class ICPFactor : public ceres::SizedCostFunction<3, 7>
{
public:
    ICPFactor(Eigen::Vector3d &point1, Eigen::Vector3d &point2, double scale = 1.0) : point1_(point1), point2_(point2), scale_(scale)
    {
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

private:
    Eigen::Vector3d point1_;
    Eigen::Vector3d point2_;
    double scale_;
};

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &q)
{
    Eigen::Matrix3d ans;
    ans << 0.0, -q(2), q(1),
        q(2), 0.0, -q(0),
        -q(1), q(0), 0.0;
    return ans;
}

bool ICPFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const

{
    Eigen::Vector3d trans(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond quat(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d res = scale_ * (quat.toRotationMatrix() * point1_ + trans - point2_);
    residuals[0] = res.x();
    residuals[1] = res.y();
    residuals[2] = res.z();
    std::cout << residuals[0] << " " << residuals[1] << " " << residuals[2] <<std::endl;

    if (jacobians)
    {

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_i(jacobians[0]);
            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = Eigen::Matrix3d::Identity();
            jaco_i.rightCols<3>() =  -quat.toRotationMatrix() * skewSymmetric(point1_); //这里千万要注意顺序，不能写反了
            
            jacobian_i.leftCols<6>() = scale_ * jaco_i;
            jacobian_i.rightCols<1>().setZero();
            cout << jacobian_i << endl;
        }
    }

    return true;
}

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
    Eigen::Vector3d t12(0.3, 0.4, 0.5);
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
        cout << "point1 " << point << endl;
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

    double q[7] = {0., 0., 0.0, 0., 0., 0., 1.};

    ceres::Problem problem;
    //problem.AddParameterBlock(q + 4, 3);

    if (GenerateSimData(points1, points2))
    {
        Eigen::Matrix4d trans = CaculateTransformation(points2, points1);
        cout << "transformation result " << trans << endl;

        double scale = 1./sqrt(points1.size());
        for (size_t i = 0; i < points1.size(); ++i)
        {
            //ICPFactor *cost_function = new ICPFactor(points1[i], points2[i], scale);
            Eigen::Vector2d point2d(points2[i].x()/points2[i].z(), points2[i].y()/points2[i].z());
            PnPFactor* cost_function = new PnPFactor(points1[i],  point2d);
            problem.AddResidualBlock(cost_function, NULL, q);
        }
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();

        problem.AddParameterBlock(q, 7, local_parameterization);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = 100;

        //options.minimizer_progress_to_stdout = true;
        // options.max_solver_time_in_seconds = 0.2;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
        Eigen::Quaterniond quat = Eigen::Quaterniond(q[6], q[3], q[4], q[5]);
        Eigen::Map<Eigen::Quaterniond> qq(q + 3);
        cout << "qq " << qq.toRotationMatrix() << endl;
        Eigen::Vector3d translation(q);
        cout << "q " << quat.toRotationMatrix() << endl;
        cout << "t " << translation << endl;


    }
    else
    {
        cout << "generate sim data error! " << endl;
    }

    return 0;
}