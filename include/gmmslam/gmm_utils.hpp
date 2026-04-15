#pragma once
#include "gmmslam/types.hpp"
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace gmmslam {

GmmModel filterWellConditioned(const GmmModel& model, double reg = 1e-4);

std::vector<GmmLocalData> precomputeGmmLocalData(const GmmModel& model);

GmmModel mergeGmmsConcatenate(
    const std::vector<std::pair<GmmModel, Matrix4d>>& gmms_with_poses,
    const Matrix4d& T_ref);

void saveGmmToFile(const GmmModel& model, const std::string& filepath);

const std::vector<std::array<double, 3>>& submapColors();

std::array<double, 3> submapColor(int sid);

} // namespace gmmslam
