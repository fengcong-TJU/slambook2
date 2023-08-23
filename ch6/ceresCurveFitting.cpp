//
// Created by xiang on 18-11-19.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;



// 代价函数的计算模型
struct CURVE_FITTING_COST {
  CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

  // 残差的计算
  template<typename T>
  bool operator()(
    const T *const abc, // 模型参数，有3维
    T *residual) const {
    residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-exp(ax^2+bx+c)
    return true;
  }

  const double _x, _y;    // x,y数据
};


//上面是三维输入一维输出的costfunction，下面是三维输入，多维输入的costfunction

struct CURVE_FITTING_COST_faster {
  CURVE_FITTING_COST_faster(vector<double> x, vector<double> y, const int N) : _x(x), _y(y) ,_N(N) {}

  // 残差的计算
  template<typename T>
  bool operator()(
    const T *const abc, // 模型参数，有3维
    T *residual) const {
      for (int i=0;i<_N;i++){
        residual[i] = T(_y[i]) - ceres::exp(abc[0] * T(_x[i]) * T(_x[i]) + abc[1] * T(_x[i]) + abc[2]); // y-exp(ax^2+bx+c)
      }   
    return true;
  }

  const vector<double> _x, _y;    // x,y数据
  const int _N;

};

int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
  double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
  const int N = 100;                                 // 数据点
  double w_sigma = 1.0;                        // 噪声Sigma值
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;                                 // OpenCV随机数产生器

  vector<double> x_data, y_data;      // 数据
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
  }

  double abc1[3] = {ae, be, ce};
  double abc2[3] = {ae, be, ce};

  
  // 构建最小二乘问题(三维输入一维输出的100个方程)
  ceres::Problem problem1;
  for (int i = 0; i < N; i++) {
    problem1.AddResidualBlock(     // 向问题中添加误差项
      // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
      new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
        new CURVE_FITTING_COST(x_data[i], y_data[i])
      ),
      nullptr,            // 核函数，这里不使用，为空
      abc1                 // 待估计参数
    );
  }

  // 构建最小二乘问题(三维输入，100维输出的一个方程)
  ceres::Problem problem2;
  problem2.AddResidualBlock(     // 向问题中添加误差项
    // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
    new ceres::AutoDiffCostFunction<CURVE_FITTING_COST_faster, N, 3>(
      new CURVE_FITTING_COST_faster(x_data, y_data,N)
    ),
    nullptr,            // 核函数，这里不使用，为空
    abc2                 // 待估计参数
  );

  // 配置求解器
  ceres::Solver::Options options;     // 这里有很多配置项可以填
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
  options.minimizer_progress_to_stdout = true;   // 输出到cout

  ceres::Solver::Summary summary;                // 优化信息
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem1, &summary);  // 开始优化
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // 输出结果
  cout << summary.BriefReport() << endl;
  cout << "estimated a,b,c = ";
  for (auto a:abc1) cout << a << " ";
  cout << endl;

  
  t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem2, &summary);  // 开始优化
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // 输出结果
  cout << summary.BriefReport() << endl;
  cout << "estimated a,b,c = ";
  for (auto a:abc2) cout << a << " ";
  cout << endl;

  return 0;
}