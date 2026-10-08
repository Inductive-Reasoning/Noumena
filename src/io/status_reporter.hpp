// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <cctype>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>
#include <nlohmann/json.hpp>

class StatusReporter {
public:
	enum class Format {
		Text,
		JsonLines
	};

	enum class Verbosity {
		Status = 0,
		Solver = 1,
		Diagnostics = 2
	};

private:
	class MessageBuffer : public std::streambuf {
	public:
		explicit MessageBuffer(StatusReporter& reporter) : reporter_(reporter) {}

	protected:
		int overflow(int ch) override;

		std::streamsize xsputn(const char* data, std::streamsize size) override;

		int sync() override {
			FlushLine();
			return 0;
		}

	private:
		void Append(char ch);

		void FlushLine();

		StatusReporter& reporter_;
		std::string line_;
	};

public:

	class Operation {
	public:
		Operation(StatusReporter& reporter, std::string name)
			: reporter_(reporter),
			  name_(std::move(name)),
			  start_(Clock::now()),
			  exception_count_(std::uncaught_exceptions()) {
			reporter_.Write("Starting", name_);
		}

		Operation(const Operation&) = delete;
		Operation& operator=(const Operation&) = delete;
		Operation(Operation&&) = delete;
		Operation& operator=(Operation&&) = delete;

		~Operation() noexcept;

	private:
		using Clock = std::chrono::steady_clock;

		StatusReporter& reporter_;
		std::string name_;
		Clock::time_point start_;
		int exception_count_;
	};

	explicit StatusReporter(std::ostream& output = std::cout,
		std::ostream& error_output = std::cerr)
		: output_(output),
		  error_output_(error_output),
		  solver_buffer_(*this),
		  solver_output_(&solver_buffer_) {}

	static StatusReporter& Global() {
		static StatusReporter reporter;
		return reporter;
	}

	static Verbosity VerbosityFromInt(int value);

	void SetVerbosity(Verbosity verbosity) { verbosity_ = verbosity; }
	Verbosity GetVerbosity() const { return verbosity_; }
	void SetFormat(Format format) { format_ = format; }
	Format GetFormat() const { return format_; }
	bool IsMachineReadable() const { return format_ == Format::JsonLines; }

	bool SolverOutputEnabled() const {
		return verbosity_ >= Verbosity::Solver;
	}

	bool DiagnosticsEnabled() const {
		return verbosity_ >= Verbosity::Diagnostics;
	}

	int SolverPrintLevel(int configured_level) const {
		return SolverOutputEnabled() ? configured_level : 0;
	}

	Operation Start(std::string name) {
		return Operation(*this, std::move(name));
	}

	void Status(const std::string& message) {
		WriteMessage("status", message);
	}

	void Diagnostic(const std::string& message);

	void Warning(const std::string& message) {
		WriteMessage("warning", message, true);
	}

	void Error(const std::string& message) {
		WriteMessage("error", message, true);
	}

	void SolverMessage(const std::string& message) {
		WriteMessage("solver", message);
	}

	std::ostream& SolverOutput() { return solver_output_; }

private:
	void WriteJson(const nlohmann::json& event) {
		output_ << event.dump() << '\n' << std::flush;
	}

	void WriteMessage(const char* level, const std::string& message,
		bool use_error_output = false);

	void Write(const char* state, const std::string& name) noexcept;

	void Write(const char* state, const std::string& name, double seconds) noexcept;

	std::ostream& output_;
	std::ostream& error_output_;
	Verbosity verbosity_ = Verbosity::Status;
	Format format_ = Format::Text;
	MessageBuffer solver_buffer_;
	std::ostream solver_output_;
};
