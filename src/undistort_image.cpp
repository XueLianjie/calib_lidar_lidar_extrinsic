#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./distorted.png"; // 请确保路径正确

int main(int argc, char **argv)
{

    // 本程序实现去畸变部分的代码。尽管我们可以调用OpenCV的去畸变，但自己实现一遍有助于理解。
    // 畸变参数
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // 内参
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat image = cv::imread(image_file, 0); // 图像是灰度图，CV_8UC1
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1); // 去畸变以后的图

    // 计算去畸变后图像的内容
    for (int v = 0; v < rows; v++)
    {
        for (int u = 0; u < cols; u++)
        {
            // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)
            double x = (u - cx) / fx, y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            // 赋值 (最近邻插值)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows)
            {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int)v_distorted, (int)u_distorted);
            }
            else
            {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }
    }

    // 画图去畸变后图像
    cv::imshow("distorted", image);
    cv::waitKey();

   // cv::imshow("undistorted", image_undistort);

    // cv::Mat lines;
    // cv::Ptr<cv::LineSegmentDetector> detector;
    // detector = cv::createLineSegmentDetector();
    // detector->detect(image_undistort, lines);
    // detector->drawSegments(image_undistort, lines);

    cv::imshow("input", image_undistort);

    cv::waitKey();
    int kernel_size = 5;
    cv::Mat blur_gray;
    cv::GaussianBlur(image_undistort, blur_gray, cv::Size(kernel_size, kernel_size), 0);
    cv::imshow("blur", blur_gray);
    cv::waitKey(0);

    float high_threshold = 150;
    float low_threshold = 50;
    cv::Mat edges;
    cv::Canny(blur_gray, edges, 50, 200);
    cv::imshow("edges", edges);
    cv::waitKey(0);

    // rho = 1  # distance resolution in pixels of the Hough grid
    // theta = np.pi / 180  # angular resolution in radians of the Hough grid
    // threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    // min_line_length = 50  # minimum number of pixels making up a line
    // max_line_gap = 20  # maximum gap in pixels between connectable line segments
    // line_image = np.copy(img) * 0  # creating a blank to draw lines on
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 105, 10, 250);
    cv::Mat image_color;
    cv::cvtColor(image_undistort,image_color,cv::COLOR_GRAY2BGR);
    for(size_t i = 0; i < lines.size(); ++i)
    {
        cv::Vec4i l = lines[i];
        cv::line(image_color, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

    }
    cv::imshow("lines", image_color);
    cv::waitKey(0);

    return 0;
}