#pragma once

#include "gmmslam/config.hpp"
#include "gmmslam/types.hpp"

namespace gmmslam {

bool gmmapFittingAvailable();

GmmModel fitGmmap(const OrganizedDepthImage& depth,
                  const SogmmConfig& config);

} // namespace gmmslam
