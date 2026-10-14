// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <string>

// Redirects std::cerr, where the status reporter writes warnings, for its
// lifetime.
class StderrCapture {
public:
	StderrCapture() : previous(std::cerr.rdbuf(text.rdbuf())) {}
	~StderrCapture() { std::cerr.rdbuf(previous); }
	StderrCapture(const StderrCapture&) = delete;
	StderrCapture& operator=(const StderrCapture&) = delete;
	std::string Text() const { return text.str(); }

private:
	std::ostringstream text;
	std::streambuf* previous;
};
