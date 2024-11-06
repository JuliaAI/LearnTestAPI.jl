# This file defines `Ensemble(atom; rng=Random.default_rng(), n=10)`

using LearnAPI
using LinearAlgebra
using Random
using Statistics


# # ENSEMBLE OF REGRESSORS (A MODEL WRAPPER)

# We implement a learner that creates an bagged ensemble of regressors, i.e, where each
# atomic model is trained on a random sample of the training observations (same number,
# but sampled with replacement). In particular this learner has an iteration parameter
# `n`, and we implement `update` to execute a warm restarts when `n` increases.

struct Ensemble
    atom # the base regressor being bagged
    rng
    n::Int
end

# Since the `atom` hyperparameter is another learner, the user must explicitly set it in
# constructor calls or an error is thrown. We also need to overload the
# `LearnAPI.is_composite` trait (done later).

"""
    Ensemble(atom; rng=Random.default_rng(), n=10)

Instantiate a bagged ensemble of `n` regressors, with base regressor `atom`, etc

"""
Ensemble(atom; rng=Random.default_rng(), n=10) =
    Ensemble(atom, rng, n) # `LearnAPI.constructor` defined later

# need a pure keyword argument constructor:
function Ensemble(; atom=nothing, kwargs...)
    isnothing(atom) && error("You must specify `atom=...` ")
    Ensemble(atom; kwargs...)
end

struct EnsembleFitted
    learner::Ensemble
    atom::Ridge
    rng    # mutated copy of `learner.rng`
    models # leaving type abstract for simplicity
end

LearnAPI.learner(model::EnsembleFitted) = model.learner

# We add the same data front end that the atomic regressor uses:
LearnAPI.obs(learner::Ensemble, data) = LearnAPI.obs(learner.atom, data)
LearnAPI.obs(model::EnsembleFitted, data) = LearnAPI.obs(first(model.models), data)
LearnAPI.target(learner::Ensemble, data) = LearnAPI.target(learner.atom, data)
LearnAPI.features(learner::Ensemble, data) = LearnAPI.features(learner.atom, data)

function LearnAPI.fit(learner::Ensemble, data; verbosity=1)

    # unpack hyperparameters:
    atom = learner.atom
    rng = deepcopy(learner.rng) # to prevent mutation of `learner`!
    n = learner.n

    # ensure data can be subsampled using MLUtils.jl, and that we're feeding the atomic
    # `fit` data in an efficient (pre-processed) form:

    observations = obs(atom, data)

    # initialize ensemble:
    models = []

    # get number of observations:
    N = MLUtils.numobs(observations)

    # train the ensemble:
    for _ in 1:n
        bag = rand(rng, 1:N, N)
        data_subset = MLUtils.getobs(observations, bag)
        # step down one verbosity level in atomic fit:
        model = fit(atom, data_subset; verbosity=verbosity - 1)
        push!(models, model)
    end

    # make some noise, if allowed:
    verbosity > 0 && @info "Trained $n ridge regression models. "

    return EnsembleFitted(learner, atom, rng, models)

end

# Consistent with the documented `update` contract, we implement this behaviour: If `n` is
# increased, `update` adds new regressors to the ensemble, including any new
# hyperparameter updates (e.g, new `atom`) when computing the new atomic
# models. Otherwise, update is equivalent to retraining from scratch, with the provided
# hyperparameter updates.
function LearnAPI.update(model::EnsembleFitted, data; verbosity=1, replacements...)
    learner_old = LearnAPI.learner(model)
    learner = LearnAPI.clone(learner_old; replacements...)

    :n in keys(replacements) || return fit(learner, data; verbosity)

    n = learner.n
    Δn = n - learner_old.n
    n < 0 && return fit(model, learner; verbosity)

    atom = learner.atom
    observations = obs(atom, data)
    N = MLUtils.numobs(observations)

    # initialize:
    models = model.models
    rng = model.rng # as mutated in previous `fit`/`update` calls

    # add new regressors to the ensemble:
    for _ in 1:Δn
        bag = rand(rng, 1:N, N)
        data_subset = MLUtils.getobs(observations, bag)
        model = fit(atom, data_subset; verbosity=verbosity-1)
        push!(models, model)
    end

    # make some noise, if allowed:
    verbosity > 0 && @info "Trained $Δn additional ridge regression models. "

    return EnsembleFitted(learner, atom, rng, models)
end

LearnAPI.predict(model::EnsembleFitted, ::Point, data) =
    mean(model.models) do atomic_model
        predict(atomic_model, Point(), data)
    end

LearnAPI.strip(model::EnsembleFitted) = EnsembleFitted(
    model.learner,
    model.atom,
    model.rng,
    LearnAPI.strip.(model.models),
)

# learner traits (note the inclusion of `iteration_parameter`):
@trait(
    Ensemble,
    constructor = Ensemble,
    iteration_parameter = :n,
    is_composite = true,
    kinds_of_proxy = (Point(),),
    tags = ("regression", "ensembling", "iterative algorithms"),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.update),
        :(LearnAPI.predict),
   )
)

# convenience method:
LearnAPI.fit(learner::Ensemble, X, y, extras...; kwargs...) =
    fit(learner, (X, y, extras...); kwargs...)
LearnAPI.update(learner::EnsembleFitted, X, y, extras...; kwargs...) =
    update(learner, (X, y, extras...); kwargs...)
