// input: raw pointcloud
// output: deskew the pointcloud ? + Fit a SOGMM and wrap in GmmFrame

// to fit a GMM from a point cloud
#include <self_organizing_gmm/SOGMMCPU.h>    // sogmm::cpu::SOGMM<float, 3>
#include <self_organizing_gmm/SOGMMLearner.h> // sogmm::cpu::SOGMMLearner<float>
