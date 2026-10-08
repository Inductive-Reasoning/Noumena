// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "status_reporter.hpp"

int StatusReporter::MessageBuffer::overflow(int ch)  {
	if (ch == traits_type::eof()) {
		return traits_type::not_eof(ch);
	}
	Append(static_cast<char>(ch));
	return ch;
}

std::streamsize StatusReporter::MessageBuffer::xsputn(const char* data, std::streamsize size)  {
	for (std::streamsize i = 0; i < size; ++i) {
		Append(data[i]);
	}
	return size;
}

void StatusReporter::MessageBuffer::Append(char ch) {
	if (ch == '\n') {
		FlushLine();
	}
	else if (ch != '\r') {
		line_ += ch;
	}
}

void StatusReporter::MessageBuffer::FlushLine() {
	if (!line_.empty()) {
		reporter_.SolverMessage(line_);
		line_.clear();
	}
}

StatusReporter::Operation::~Operation() noexcept {
	const double seconds =
		std::chrono::duration<double>(Clock::now() - start_).count();
	reporter_.Write(std::uncaught_exceptions() > exception_count_
						? "Failed"
						: "Completed",
					name_, seconds);
}

 StatusReporter::Verbosity StatusReporter::VerbosityFromInt(int value) {
	switch (value) {
		case 0: return Verbosity::Status;
		case 1: return Verbosity::Solver;
		case 2: return Verbosity::Diagnostics;
		default:
			throw std::invalid_argument("verbosity must be 0, 1, or 2");
	}
}

void StatusReporter::Diagnostic(const std::string& message) {
	if (DiagnosticsEnabled()) {
		WriteMessage("diagnostic", message);
	}
}

void StatusReporter::WriteMessage(const char* level, const std::string& message,
	bool use_error_output) {
	if (IsMachineReadable()) {
		WriteJson({
			{"event", "message"},
			{"level", level},
			{"message", message}
		});
	}
	else {
		std::ostream& stream = use_error_output ? error_output_ : output_;
		stream << message << '\n' << std::flush;
	}
}

void StatusReporter::Write(const char* state, const std::string& name) noexcept {
	try {
		if (IsMachineReadable()) {
			WriteJson({
				{"event", "operation"},
				{"state", state == std::string("Starting") ? "started" : state},
				{"name", name}
			});
		}
		else {
			output_ << state << ' ' << name << "...\n" << std::flush;
		}
	}
	catch (...) {
	}
}

void StatusReporter::Write(const char* state, const std::string& name, double seconds) noexcept {
	try {
		if (IsMachineReadable()) {
			std::string machine_state = state;
			machine_state[0] = static_cast<char>(std::tolower(
				static_cast<unsigned char>(machine_state[0])));
			WriteJson({
				{"event", "operation"},
				{"state", machine_state},
				{"name", name},
				{"elapsed_seconds", seconds}
			});
		}
		else {
			std::ostringstream elapsed;
			elapsed << std::fixed << std::setprecision(3) << seconds;
			output_ << state << ' ' << name << " in " << elapsed.str() << " s\n"
				<< std::flush;
		}
	}
	catch (...) {
	}
}
