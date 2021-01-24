#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"
#include "csv.h"
#include "Eigen/Eigen"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct AxisRotationError {
    AxisRotationError(double observed_x0, double observed_y0, double observed_z0, double observed_x1, double observed_y1, double observed_z1)
        : observed_x0(observed_x0), observed_y0(observed_y0), observed_z0(observed_z0), observed_x1(observed_x1), observed_y1(observed_y1), observed_z1(observed_z1) {}

    template <typename T>
    bool operator()(const T* const axis, const T* const angle, const T* const trans, T* residuals) const {
    //bool operator()(const T* const axis, const T* const trans, T* residuals) const {
        // Normalize axis
        T a[3];
        T k = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
        a[0] = axis[0] / sqrt(k);
        a[1] = axis[1] / sqrt(k);
        a[2] = axis[2] / sqrt(k);

        // Define quaternion from axis and angle. Convert angle to radians
        T pi = T(3.14159265359);
        //T angle[1] = {T(10.0)};
        T quaternion[4] = { cos((angle[0]*pi / 180.0) / 2.0), 
            a[0] * sin((angle[0] * pi / 180.0) / 2.0),
            a[1] * sin((angle[0] * pi / 180.0) / 2.0),
            a[2] * sin((angle[0] * pi / 180.0) / 2.0) };

        // Define transformation
        T t[3] = { trans[0], trans[1], trans[2] };

        // Calculate predicted positions
        T observedPoint0[3] = { T(observed_x0), T(observed_y0), T(observed_z0)};
        T point[3]; point[0] = observedPoint0[0] - t[0]; point[1] = observedPoint0[1] - t[1]; point[2] = observedPoint0[2] - t[2];
        T rotatedPoint[3];
        ceres::QuaternionRotatePoint(quaternion, point, rotatedPoint);
        T predicted_x = rotatedPoint[0] + t[0];
        T predicted_y = rotatedPoint[1] + t[1];
        T predicted_z = rotatedPoint[2] + t[2];

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x1);
        residuals[1] = predicted_y - T(observed_y1);
        residuals[2] = predicted_z - T(observed_z1);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x0, const double observed_y0, const double observed_z0, 
                                       const double observed_x1, const double observed_y1, const double observed_z1) {
        // Define AutoDiffCostFunction. <AxisRotationError, #residuals, #dim axis, #dim angle, #dim trans
        return (new ceres::AutoDiffCostFunction<AxisRotationError, 3, 3, 1,3>(
            new AxisRotationError(observed_x0, observed_y0, observed_z0, observed_x1, observed_y1, observed_z1)));
    }

    double observed_x0;
    double observed_y0;
    double observed_z0;
    double observed_x1;
    double observed_y1;
    double observed_z1;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // Load points.csv into cv::Mat's
  // 216 rows with (x0, y0, z0, x1, y1, z1)
  // [x1,y1,z1] = R* [x0-tx,y0-ty,z0-tz] + [tx,ty,tz]
  // The xyz coordinates are points on a chessboard, where the chessboard 
  // is rotated for 4x. Each chessboard has 54 xyz points. So 4x 54,
  // gives the 216 rows of observations.
  // The chessboard is located at [0,0,1], as the camera_0 is located
  // at [-0.1,0,0], the t should become [0.1,0,1.0].
  // The chessboard is rotated around axis [0.0,1.0,0.0]
  io::CSVReader<6> in("points.csv");
  float x0, y0, z0, x1, y1, z1;

  // The observations
  cv::Mat x_0(216, 3, CV_32F);
  cv::Mat x_1(216, 3, CV_32F);
  for (int rowNr = 0; rowNr < 216; rowNr++){
      if (in.read_row(x0, y0, z0, x1, y1, z1))
      {
          x_0.at<float>(rowNr, 0) = x0;
          x_0.at<float>(rowNr, 1) = y0;
          x_0.at<float>(rowNr, 2) = z0;
          x_1.at<float>(rowNr, 0) = x1;
          x_1.at<float>(rowNr, 1) = y1;
          x_1.at<float>(rowNr, 2) = z1;
      }
  }

  std::cout << x_0(cv::Rect(0, 0, 2, 5)) << std::endl;

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  int numObservations = 216;
  double axis[3] = { 0.0, 1.0, 0.0 };
  double* pAxis; pAxis = axis;
  double angles[4] = { 10.0, 10.0, 10.0, 10.0 };
  double* pAngles; pAngles = angles;
  double t[3] = { 0.0, 0.0, 1.0,};
  double* pT; pT = t;
  bool FLAGS_robustify = true;

  // Build the problem.
  Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  for (int i = 0; i < numObservations; ++i) {
      ceres::CostFunction* cost_function =
          AxisRotationError::Create(
          x_0.at<float>(i, 0), x_0.at<float>(i, 1), x_0.at<float>(i, 2),
          x_1.at<float>(i, 0), x_1.at<float>(i, 1), x_1.at<float>(i, 2));
      //std::cout << "pAngles: " << pAngles[i / 54] << ", " << i / 54 << std::endl;
      ceres::LossFunction* loss_function = FLAGS_robustify ? new ceres::HuberLoss(0.001) : NULL;
      //ceres::LossFunction* loss_function = FLAGS_robustify ? new ceres::CauchyLoss(0.002) : NULL;
      problem.AddResidualBlock(cost_function, loss_function, pAxis, &pAngles[i/54], pT);
      //problem.AddResidualBlock(cost_function, loss_function, pAxis, pT);
  }

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.num_threads = 4;
  options.use_nonmonotonic_steps = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //std::cout << summary.FullReport() << "\n";
  std::cout << summary.BriefReport() << "\n";

  // Normalize axis
  double k = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
  axis[0] = axis[0] / sqrt(k);
  axis[1] = axis[1] / sqrt(k);
  axis[2] = axis[2] / sqrt(k);

  // Plot results
  std::cout << "axis: [ " << axis[0] << "," << axis[1] << "," << axis[2] << " ]" << std::endl;
  std::cout << "t: [ " << t[0] << "," << t[1] << "," << t[2] << " ]" << std::endl;
  std::cout << "angles: [ " << angles[0] << "," << angles[1] << "," << angles[2] << "," << angles[3] << " ]" << std::endl;

  return 0;
}