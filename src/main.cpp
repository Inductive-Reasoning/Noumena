// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "mfem.hpp"
#include "build_info.hpp"
#include "config/input_parser.hpp"
#include "solvers/physics_solver.hpp"
#include "solvers/solver_factory.hpp"
#include "config/config_validator.hpp"
#include "io/status_reporter.hpp"
#include "io/mesh_loader.hpp"
#include "parallel/mpi_runtime.hpp"

namespace {

void PrintUsage(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " <config.json> [options]\n"
        "Options:\n"
        "  --output-directory <dir>   Override output.directory. A relative\n"
        "                             path resolves against the current directory.\n"
        "  --verbosity <0|1|2>        0=status/timing, 1=solver output, 2=diagnostics\n"
        "  --machine-readable         Emit flushed JSON Lines progress on stdout\n"
        "  --version                  Print version/build information and exit\n"
        "  -h, --help                 Show this help\n";
}

// std::stoi would accept trailing garbage ("1abc" -> 1) and reports failures as
// a bare "invalid stoi argument" that never names the offending option. Parse
// strictly instead so the user is told which flag was wrong and what it accepts.
int ParseVerbosity(const std::string& text) {
    size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed);
    } catch (const std::exception&) {
        consumed = 0;
    }
    if (consumed != text.size() || value < 0 || value > 2) {
        throw std::runtime_error(
            "--verbosity must be 0, 1, or 2 (got '" + text + "')");
    }
    return value;
}

} // namespace

int main(int argc, char *argv[]) {
    // MPI must be initialized before anything else touches it; a no-op in a
    // serial build.
    parallel::Initialize(argc, argv);

    // Convert MFEM internal errors (MFEM_ASSERT / MFEM_VERIFY / MFEM_ABORT)
    // into catchable mfem::ErrorException objects so a malformed mesh does
    // not call std::abort() from deep inside the Mesh constructor. The mesh
    // loading path below depends on this, so a build without it is rejected
    // outright rather than silently degrading to abort-on-bad-mesh.
#ifndef MFEM_USE_EXCEPTIONS
#error "MFEM must be built with MFEM_USE_EXCEPTIONS=ON; see top-level CMakeLists.txt"
#endif
    mfem::set_error_action(mfem::MFEM_ERROR_THROW);
    StatusReporter& reporter = StatusReporter::Global();

    // Pre-scan for --machine-readable before the real argument loop: the output
    // format has to be fixed before anything can fail, otherwise an early error
    // (bad option, missing config) would escape as plain text and break a caller
    // parsing JSON Lines. The main loop below matches the flag again so it does
    // not trip the unknown-option check.
    bool machine_readable_requested = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--machine-readable") {
            machine_readable_requested = true;
            break;
        }
    }
    if (machine_readable_requested) {
        reporter.SetFormat(StatusReporter::Format::JsonLines);
    }

    try {
        // An MPI build runs on one rank until the solvers and writers are
        // distributed; more ranks would each repeat the whole run and race on
        // the same output files.
        if (parallel::WorldSize() > 1) {
            throw std::runtime_error("This build runs on a single MPI rank; "
                "launch with one process (mpirun -np 1) or none. Multi-rank "
                "solves are not implemented yet.");
        }

        std::string config_file = "config.json";
        std::string cli_output_directory;
        int cli_verbosity = 0;
        bool verbosity_explicit = false;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto need_value = [&](const std::string& opt) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(opt + " requires a value");
                }
                return std::string(argv[++i]);
            };

            if (a == "-h" || a == "--help") {
                PrintUsage(argv[0]);
                return 0;
            } else if (a == "--version") {
                std::cout << build_info::Describe() << '\n';
                return 0;
            } else if (a == "--output-directory") {
                cli_output_directory = need_value(a);
            } else if (a == "--verbosity") {
                cli_verbosity = ParseVerbosity(need_value(a));
                verbosity_explicit = true;
            } else if (a == "--machine-readable") {
                machine_readable_requested = true;
            } else if (!a.empty() && a[0] == '-') {
                throw std::runtime_error("Unknown option: " + a);
            } else {
                config_file = a;
            }
        }

        if (machine_readable_requested && !verbosity_explicit) {
            cli_verbosity = 1;
        }
        reporter.SetVerbosity(StatusReporter::VerbosityFromInt(cli_verbosity));
        if (machine_readable_requested) {
            mfem::out.SetStream(reporter.SolverOutput());
        }

        // Report build identity up front so a run can always be traced back to
        // a specific binary; a stale executable is otherwise only detectable by
        // the confusing downstream errors it produces. Routed through the
        // reporter so --machine-readable still emits well-formed JSON Lines.
        reporter.Status(build_info::Describe());

        // Shared Infrastructure
        std::error_code config_ec;
        auto config_abs = std::filesystem::weakly_canonical(config_file, config_ec);
        reporter.Diagnostic("Config file: "
            + (config_ec ? std::filesystem::absolute(config_file).string()
                         : config_abs.string()));

        std::unique_ptr<InputParser> parser_ptr;
        {
            auto operation = reporter.Start("configuration loading");
            parser_ptr = std::make_unique<InputParser>(config_file);
        }
        InputParser& parser = *parser_ptr;

        // Inject the CLI override into the shared json before decoding.
        // const_cast is safe: parser owns the json when constructed from a path.
        json& mutable_json = const_cast<json&>(parser.config);
        if (!mutable_json.contains("simulation") || !mutable_json["simulation"].is_object()) {
            mutable_json["simulation"] = json::object();
        }
        if (!cli_output_directory.empty()) {
            if (mutable_json.contains("output") && !mutable_json["output"].is_object()) {
                throw std::runtime_error("output must be an object");
            }
            mutable_json["output"]["directory"] =
                std::filesystem::absolute(cli_output_directory).string();
        }

        // Validate Configuration (schema and basic semantics before decoding)
        ConfigValidator validator;
        {
            auto operation = reporter.Start("configuration validation");
            validator.ValidateOrThrow(parser.config);
        }

        ProblemConfig config = parser.GetProblemConfig();

        // Load Mesh
        std::error_code mesh_ec;
        auto mesh_abs = std::filesystem::weakly_canonical(config.MeshPath, mesh_ec);
        reporter.Diagnostic("Mesh file: "
            + (mesh_ec ? std::filesystem::absolute(config.MeshPath).string()
                       : mesh_abs.string()));

        // MFEM reports a missing/unreadable file through the same error path
        // as a malformed one, so check it up front to keep the two failures
        // distinguishable for the user.
        std::error_code exists_ec;
        if (!std::filesystem::is_regular_file(config.MeshPath, exists_ec)) {
            throw std::runtime_error(
                "Mesh file not found or not readable: '" + config.MeshPath + "'");
        }

        std::unique_ptr<mfem::Mesh> mesh_ptr;
        {
            auto operation = reporter.Start("mesh loading");
            try {
                mesh_ptr = mesh_io::LoadMesh(config.MeshPath);
            }
            catch (const mfem::ErrorException& e) {
                throw std::runtime_error(
                    "MFEM could not load mesh '" + config.MeshPath + "'. Check "
                    "that the file format is supported, that elements are "
                    "consistently oriented (counter-clockwise winding in 2D), and "
                    "that boundary elements lie on element faces with shared nodes.\nMFEM detail: " + std::string(e.what()));
            }
        }

        mfem::Mesh& mesh = *mesh_ptr;

        // Validate Configuration (with mesh for attribute checking)
        {
            auto operation = reporter.Start("configuration validation");
            validator.ValidateOrThrow(parser.config, &mesh);
        }

        // Factory Logic - Create Solver
        std::unique_ptr<PhysicsSolver> solver;
        {
            auto operation = reporter.Start("solver creation");
            solver = SolverFactory::Instance().Create(mesh, config);
        }

        // Execution
        {
            auto operation = reporter.Start("solver setup");
            solver->Setup();
        }
        {
            auto operation = reporter.Start("solution process");
            solver->Run();
        }
        {
            auto operation = reporter.Start("analysis result writing");
            solver->SaveAnalysis();
        }

        return 0;
    }
    catch (const std::exception& e) {
        reporter.Error(e.what());
        return 1;
    }
    catch (...) {
        reporter.Error("Unknown exception occurred");
        return 2;
    }
}