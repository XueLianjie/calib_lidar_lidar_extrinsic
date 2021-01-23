#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;  // 真实的参数值
    double ae = 2.0, be = -1.0, ce = 5.0; // 采用迭代方法优化的参数初始值
    int N = 100;                          // 数据量
    double w_sigma = 1.0;                 // 参数的噪声大小
    double inv_sigma = 1.0 / w_sigma;     // 马氏距离协方差矩阵的逆，噪声值的倒数
    cv::RNG rng;                          // opencv 随机数生成器

    vector<double> x_data, y_data;
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }
    int iterations = 100;
    double cost = 0, lastCost = 0;

    for (int iter = 0; iter < iterations; ++iter)
    {
        Matrix3d H = Matrix3d::Zero();
        Vector3d b = Vector3d::Zero();
        cost = 0;

        for (int i = 0; i < N; ++i)
        {

            double xi = x_data[i], yi = y_data[i];
            double error = yi - exp(ae * xi * xi + be * xi + ce);
            Vector3d J;
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);
            J[2] = -exp(ae * xi * xi + be * xi + ce);
            H += inv_sigma * inv_sigma * J * J.transpose(); //这个权重非常重要，不同的参数项需要设置不同的权重，尤其是在传感器融合的时候这里非常重要
            b += -inv_sigma * error * J;
            cost += error * error;
        }
        Vector3d dx = H.ldlt().solve(b); //线性求解出deltax
        if (isnan(dx[0]))
        {
            cout << "result is nan ! " << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost)
        {
            cout << "error" << endl;
        }
        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        lastCost = cost;
        cout << "total cost : " << cost << ", \t \tupdata: " << dx.transpose() << endl;
    }

    cout << ae << " " << be << " " << ce << std::endl;

    return 0;
}
