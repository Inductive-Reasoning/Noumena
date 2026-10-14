// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "mfem.hpp"
#include "problem_config.hpp"

// A boundary condition paired with the mesh marker it resolves to. The marker is
// computed once (attribute values are refinement-invariant) and reused.
//
// Every instance corresponds to a prescribed entry in config.BoundaryConditions.
// Terminals are deliberately NOT represented here: a terminal is a model-level
// connection to the outside world, and only one of its realizations (an
// electrostatic voltage) is a boundary condition -- current terminals resolve to
// domain sources or to coupled global unknowns. See
// docs/boundary_and_terminal_model.md.
struct MarkedBoundaryCondition {
	mfem::Array<int> Marker;
	BoundaryCondition Condition;

	bool IsDirichlet() const { return Condition.Type == BoundaryConditionType::Dirichlet; }
	bool IsNeumann() const { return Condition.Type == BoundaryConditionType::Neumann; }
	bool IsRobin() const { return Condition.Type == BoundaryConditionType::Robin; }

	// Carries a fixed nonzero value that must be re-projected each scenario:
	// FormLinearSystem lifts essential values into the RHS, and the solution
	// vector is reset between scenarios.
	bool IsNonzeroDirichlet() const { return IsDirichlet() && Condition.Value != 0.0; }
};

// OR a marker into an accumulating one. Essential DOFs come from several
// independent sources (Dirichlet conditions, voltage terminals, axis
// regularity); each solver unions the ones its formulation has.
inline void MergeMarker(mfem::Array<int>& target, const mfem::Array<int>& source) {
	const int n = std::min(target.Size(), source.Size());
	for (int i = 0; i < n; ++i) {
		if (source[i]) { target[i] = 1; }
	}
}

// The prescribed boundary conditions of one solve
// the solvers and validation need. Owning the folds here keeps "which boundary
// conditions are Dirichlet?" a single question with a single answer.
class BoundaryConditionSet {
	std::vector<MarkedBoundaryCondition> entries;

public:
	void Add(mfem::Array<int> marker, const BoundaryCondition& condition) {
		entries.push_back({ std::move(marker), condition });
	}

	bool empty() const { return entries.empty(); }
	auto begin() const { return entries.begin(); }
	auto end() const { return entries.end(); }

	const std::vector<MarkedBoundaryCondition>& Entries() const { return entries; }

	// Marker (1/0 over bdr attributes) of the Dirichlet conditions in this set.
	// This is THIS SET's contribution to ess_bdr, not the finished article: the
	// caller unions in terminals and axis regularity as its formulation requires.
	mfem::Array<int> DirichletMarker(int max_bdr_attr) const {
		mfem::Array<int> marker(max_bdr_attr);
		marker = 0;
		for (const auto& bc : entries) {
			if (bc.IsDirichlet()) { MergeMarker(marker, bc.Marker); }
		}
		return marker;
	}
};
