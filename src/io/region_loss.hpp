// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

/// The time-averaged Joule loss [W] of one conducting region (a massive
/// terminal, a material region, or an unnamed conducting attribute).
struct RegionLoss {
	std::string Name;
	double Power = 0.0;
};
