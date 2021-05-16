#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

float rescale_sigma(float linear_sigma) {
  return 0.82357472f * std::exp(0.68797398f * linear_sigma);
}

struct a_trous_filter {
  std::vector<int> x_idxs;
  std::vector<int> y_idxs;
  std::vector<float> weights;
};

int main() {
  auto input_img = cv::imread("HSeq_londonbridge_1.ppm", 0);
  cv::Mat float_input_img;
  input_img.convertTo(float_input_img, CV_32FC1);

  const size_t scale_depth = 3;
  const float extrema_threshold = 0.05f;
  const float cm_lower_threshold = 0.7f;
  const float cm_upper_threshold = 1.5f;

  std::array<a_trous_filter, scale_depth + 3> h_filters;
  {
    const cv::Mat gauss_kernel = cv::getGaussianKernel(5, 0.6, CV_32F);
    for (int y = -2; y <= 2; y++) {
      for (int x = -2; x <= 2; x++) {
        h_filters[0].x_idxs.push_back(x);
        h_filters[0].y_idxs.push_back(y);
        h_filters[0].weights.push_back(gauss_kernel.at<float>(2 + x, 0) *
                                       gauss_kernel.at<float>(2 + y, 0) / 255.f);
      }
    }

    const float base_spline[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
    for (size_t i = 1; i <= scale_depth + 2; i++) {
      const int pitch = (int)std::pow(2, i - 1) - 1;
      const int half_size = (5 + 4 * pitch) / 2;

      for (int yi = -2; yi <= 2; yi++) {
        for (int xi = -2; xi <= 2; xi++) {
          h_filters[i].x_idxs.push_back((pitch + 1) * xi);
          h_filters[i].y_idxs.push_back((pitch + 1) * yi);
          h_filters[i].weights.push_back(base_spline[2 + xi] * base_spline[2 + yi]);
        }
      }
    }
  }

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  std::array<cv::Mat, scale_depth + 3> coarse_imgs;
  cv::Mat prev_img = float_input_img;
  for (size_t d = 0; d < scale_depth + 3; d++) {
    coarse_imgs[d] = cv::Mat::zeros(input_img.rows, input_img.cols, CV_32F);
    for (size_t y = 0; y < input_img.rows; y++) {
      for (size_t x = 0; x < input_img.cols; x++) {
        for (size_t i = 0; i < 25; i++) {
          int tx = x + h_filters[d].x_idxs[i];
          int ty = y + h_filters[d].y_idxs[i];
          float w = h_filters[d].weights[i];
          // if (ty < 0) {
          //   ty = -ty;
          // } else if (ty >= prev_img.rows) {
          //   ty = 2 * prev_img.rows - ty;
          // }
          // if (tx < 0) {
          //   tx = -tx;
          // } else if (tx >= prev_img.cols) {
          //   tx = 2 * prev_img.cols - tx;
          // }
          if (0 <= ty && ty < prev_img.rows && 0 <= tx && tx < prev_img.cols) {
            coarse_imgs[d].at<float>(y, x) += w * prev_img.at<float>(ty, tx);
          }
        }
      }
    }
    prev_img = coarse_imgs[d];
  }

  end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << elapsed << "[ms]" << std::endl;

  start = std::chrono::system_clock::now();

  std::array<cv::Mat, scale_depth + 2> fine_imgs;
  {
    for (size_t i = 0; i < scale_depth + 2; i++) {
      fine_imgs[i] = coarse_imgs[i] - coarse_imgs[i + 1];
    }

    // for (size_t i = 0; i < scale_depth + 2; i++) {
    //   double minVal;
    //   double maxVal;
    //   cv::Point minLoc;
    //   cv::Point maxLoc;

    //   cv::minMaxLoc(fine_imgs[i], &minVal, &maxVal, &minLoc, &maxLoc);

    //   std::cout << "max val: " << maxVal << std::endl;
    // }
  }

  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << elapsed << "[ms]" << std::endl;

  cv::Mat dx_kernel = (cv::Mat_<float>(1, 3) << -0.5f, 0, 0.5f);
  cv::Mat dy_kernel = (cv::Mat_<float>(3, 1) << -0.5f, 0, 0.5f);
  cv::Mat hxx_kernel = (cv::Mat_<float>(1, 3) << 1.f, -2.f, 1.f);
  cv::Mat hyy_kernel = (cv::Mat_<float>(3, 1) << 1.f, -2.f, 1.f);

  start = std::chrono::system_clock::now();

  std::vector<cv::KeyPoint> keypoints;
  for (size_t d = 0; d < scale_depth; d++) {
    cv::Mat &pre = fine_imgs[d];
    cv::Mat &cur = fine_imgs[d + 1];
    cv::Mat &nxt = fine_imgs[d + 2];

    for (size_t y = 1; y < input_img.rows - 1; y++) {
      for (size_t x = 1; x < input_img.cols - 1; x++) {
        if (cur.at<float>(y, x) < extrema_threshold - 0.01) continue;

        cv::Mat dD(3, 1, CV_32F);
        dD.at<float>(0, 0) = (cur.at<float>(y, x + 1) - cur.at<float>(y, x - 1)) / 2.f;
        dD.at<float>(1, 0) = (cur.at<float>(y + 1, x) - cur.at<float>(y - 1, x)) / 2.f;
        dD.at<float>(2, 0) = (nxt.at<float>(y, x) - pre.at<float>(y, x)) / 2.f;

        cv::Mat H(3, 3, CV_32F);
        H.at<float>(0, 0) =
            cur.at<float>(y, x + 1) + cur.at<float>(y, x - 1) - 2.f * cur.at<float>(y, x);
        H.at<float>(1, 1) =
            cur.at<float>(y + 1, x) + cur.at<float>(y - 1, x) - 2.f * cur.at<float>(y, x);
        H.at<float>(2, 2) = nxt.at<float>(y, x) + pre.at<float>(y, x) - 2.f * cur.at<float>(y, x);
        H.at<float>(0, 1) = H.at<float>(1, 0) =
            (cur.at<float>(y + 1, x + 1) - cur.at<float>(y + 1, x - 1) -
             cur.at<float>(y - 1, x + 1) + cur.at<float>(y - 1, x - 1)) /
            4.f;
        H.at<float>(0, 2) = H.at<float>(2, 0) =
            (nxt.at<float>(y, x + 1) - pre.at<float>(y, x + 1) - nxt.at<float>(y, x - 1) +
             pre.at<float>(y, x - 1)) /
            4.f;
        H.at<float>(1, 2) = H.at<float>(2, 1) =
            (nxt.at<float>(y + 1, x) - pre.at<float>(y + 1, x) - nxt.at<float>(y - 1, x) +
             pre.at<float>(y - 1, x)) /
            4.f;

        cv::Mat dpos;
        cv::solve(H, -dD, dpos);
        if (std::abs(dpos.at<float>(0, 0)) < 0.5f && std::abs(dpos.at<float>(1, 0)) < 0.5f &&
            std::abs(dpos.at<float>(2, 0)) < 0.5f) {
          float response = cur.at<float>(y, x) + dD.dot(dpos) / 2.f;
          cv::KeyPoint kpt(cv::Point2f(x + dpos.at<float>(0, 0), y + dpos.at<float>(1, 0)),
                           rescale_sigma(d + 1 + dpos.at<float>(2, 0)), response = response);
          keypoints.push_back(kpt);
        }
      }
    }
  }
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << elapsed << "[ms]" << std::endl;
  std::cout << keypoints.size() << std::endl;
}

