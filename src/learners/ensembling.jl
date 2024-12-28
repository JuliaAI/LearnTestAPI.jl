# This file defines:

# - `Ensemble(atom; rng=Random.default_rng(), n=10)` (a model wrapper)
# - `StumpRegressor(; fraction_train=0.8, ntrees=10, rng=...)`

using LearnAPI
using LinearAlgebra
using Random
using Statistics
using UnPack


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
    Δn < 0 && return fit(model, learner; verbosity)

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


# # ENSEMBLE OF EXTREMELY RANDOMIZED TREE STUMP REGRESSORS

# This simplistic tree-ensembler allows the user to specify a split of the training data
# into internal "train" and "validation" sets; something like this is common in gradient
# boosting algorithms. The ensembler does not use the validation set for early stopping
# (the typical use case) but instead exposes the internally computed predictions on the
# validation (and train) sets to allow the user (or some meta-algorithm) to externally
# control the iteration. While one could also compute these predictions externally, the
# ensembler can do so more efficiently, by progressively updating them as each new tree is
# added to the ensemble. This is the point.

# See /test/ensembling.jl for a demonstration of external iteration control using the
# exposed internal predictions.

# This learner has an iteration parameter `ntrees`, and we implement `update` to execute a
# warm restarts when `ntrees` increases.

# The following LearnAPI functions are implementened:

# - `fit`
# - `update`
# - `predict` (`Point` predictions)
# - `predictions` (returns predictions on all supplied data)
# - `out_of_sample_indices` (articluates which data is the internal validation data)
# - `trees`
# - `training_losses`
# - `out_of_sample_losses`

# For simplicity, this implementation is restricted to univariate features. The simplistic
# algorithm is explained in the docstring.  of the data presented.


# ## HELPERS

# decision stump:
struct Stump
    splitting_value::Float64
    left_target::Float64
    right_target::Float64
end
_predict(stump::Stump, ξ) =
    ξ < stump.splitting_value ? stump.left_target : stump.right_target
_predict(stump::Stump, x::AbstractVector) = _predict.(Ref(stump), x)
_predict(forest::Vector{Stump}, x) = _predict.(forest, Ref(x)) |> mean

function splitting_value(rng, xmin, xmax)
    t = rand(rng)
    return (1 - t)*xmin + t*xmax
end

function left_right(ξ, x, y)
    left = 0
    right = 0
    left_count = 0
    right_count = 0
    for i in eachindex(x)
        if x[i] < ξ
            left += y[i]
            left_count += 1
        else
            right += y[i]
            right_count += 1
        end
    end
    return (left/left_count, right/right_count)
end


# ## METHOD TO ADD A STUMP TO A FOREST

# helper to create a new `Stump` and add it to the a forest, and to update internal
# predictions (on all data) and losses:
function update!(
    forest,                 # vector of `Stump`s
    predictions,            # vector of target predictions on all data
    training_losses,        # vector, one element per iteration
    out_of_sample_losses,   # vector, one element per iteration
    training_indices,
    out_of_sample_indices,
    xmin,    # min(xtrain...)
    xmax,    # max(xtrain...)
    x,       # all the features
    xtrain,  # training features
    ytrain,  # training target
    yvalid,  # validation (out-of-sample) target
    rng,
    verbosity,
    )
    k = length(forest)
    ξ = splitting_value(rng, xmin, xmax)
    left, right = left_right(ξ, xtrain, ytrain)
    stump = Stump(ξ, left, right)
    push!(forest, stump)
    new_predictions = _predict(stump, x)
    # efficient in-place update of `predictions`:
    predictions .= (k*predictions .+ new_predictions)/(k + 1)
    push!(training_losses, (predictions[training_indices] .- ytrain).^2 |> sum)
    out_of_sample_loss = isempty(out_of_sample_indices) ? Inf :
        (predictions[out_of_sample_indices] .- yvalid).^2 |> sum
    push!(out_of_sample_losses, out_of_sample_loss)
    if !isempty(out_of_sample_indices) && verbosity > 1
        @info "out_of_sample_loss: $out_of_sample_loss"
    end
    return nothing
end


# ## IMPLEMENTATION

struct StumpRegressor
    ntrees::Int
    fraction_train::Float64
    rng
end

"""
    StumpRegressor(; ntrees=10, fraction_train=0.8, rng=Random.default_rng())

Instantiate `StumpRegressor` learner for training using the LearnAPI.jl interface, as in
the example below. By default, 20% of the data is internally set aside to allow for
tracking an out-of-sample loss. Internally computed predictions (on the full data) are
also exposed to the user.

```
x = rand(100)
y = sin.(x)

learner = LearnTestAPI.StumpRegressor(ntrees=100)

# train regressor with 100 tree stumps, printing running out-of-sample loss:
model = fit(learner, (x, y), verbosity=2)

# add 400 stumps:
model = update(model, (x, y), ntrees=500)

# predict:
@assert predict(model, x) ≈ LearnAPI.predictions(model)

# inspect other byproducts of training:
LearnAPI.training_losses(model)
LearnAPI.out_of_sample_losses(model)
LearnAPI.trees(model)

```

Only univariate data is supported. Data is cached and `update(model, data; ...)` ignores
`data`.

# Algorithm

Predictions in this simplistic algorithm are averages over an ensemble of decision tree
stumps. Each new stump has it's feature split threshold chosen uniformly at random,
between the minimum and maximum values present in the training data. Predictions on new
feature values to the left (resp., right) of the threshold mean values of the target
for training observations in which the feature is less than (resp., greater than) the
threshold. This algorithm is not intended for practical application.

"""
StumpRegressor(; ntrees=10, fraction_train=0.8, rng=Random.default_rng()) =
    StumpRegressor(ntrees, fraction_train, rng)

struct StumpRegressorFitted
    learner::StumpRegressor
    forest::Vector{Stump}
    predictions::Vector{Float64}
    training_losses::Vector{Float64}
    out_of_sample_losses::Vector{Float64}
    training_indices::Vector{Int}
    out_of_sample_indices::Vector{Int}
    xmin::Float64
    xmax::Float64
    x::Vector{Float64}
    xtrain::Vector{Float64}
    ytrain::Vector{Float64}
    yvalid::Vector{Float64}
    rng
end

function LearnAPI.fit(learner::StumpRegressor, data; verbosity=LearnAPI.default_verbosity())

    x, y = data
    rng = deepcopy(learner.rng)

    # split the data:
    n = length(y)
    ntrain = round(Int, learner.fraction_train*n)
    training_indices = 1:ntrain
    out_of_sample_indices = (ntrain + 1):n
    xtrain = x[training_indices]
    ytrain = y[training_indices]
    yvalid = y[out_of_sample_indices]

    # compute feature range
    xmin = min(xtrain...)
    xmax = max(xtrain...)

    # first iteration:
    ξ = splitting_value(rng, xmin, xmax)
    left, right = left_right(ξ, xtrain, ytrain)
    forest = [Stump(ξ, left, right),]
    predictions = _predict(forest, x)
    training_losses = [(predictions[training_indices] .- ytrain).^2 |> sum, ]
    out_of_sample_loss = isempty(out_of_sample_indices) ? Inf :
            (predictions[out_of_sample_indices] .- yvalid).^2 |> sum
    out_of_sample_losses = [out_of_sample_loss,]
    if !isempty(out_of_sample_indices) && verbosity > 1
        @info "out_of_sample_loss: $out_of_sample_loss"
    end

    args = (
        forest,
        predictions,
        training_losses,
        out_of_sample_losses,
        training_indices,
        out_of_sample_indices,
        xmin,
        xmax,
        x,
        xtrain,
        ytrain,
        yvalid,
        rng,
    )

    # complete the ensemble:
    for _ in 2:learner.ntrees
        update!(args..., verbosity)
    end

    return StumpRegressorFitted(learner, args...)

end
function Base.show(io::IO, ::MIME"text/plain", model::StumpRegressorFitted)
    println(io, "StumpRegressorFitted:")
    println(io, "  ntrees: $(model.learner.ntrees)")
    print(io, "  out_of_sample_loss: $(last(model.out_of_sample_losses))")
end

function LearnAPI.update(
    model::StumpRegressorFitted,
    data; # ignored as cached
    verbosity=LearnAPI.default_verbosity(),
    replacements...,
    )

    learner_old = LearnAPI.learner(model)

    @unpack learner,
    forest,
    predictions,
    training_losses,
    out_of_sample_losses,
    training_indices,
    out_of_sample_indices,
    xmin,
    xmax,
    x,
    xtrain,
    ytrain,
    yvalid,
    rng = model

    learner_old = learner
    learner = LearnAPI.clone(learner_old; replacements...)

    kys = keys(replacements)
    :ntrees in kys || return fit(learner, data; verbosity)
    # other replacements are ignored:
    for k in kys
        if k != :ntrees && verbosity ≥ 0
            @warn "Ignoring the change to `$k`. "
        end
    end

    ntrees = learner.ntrees
    Δntrees = ntrees - learner_old.ntrees
    Δntrees < 0 && return fit(model, learner; verbosity)

        args = (
        forest,
        predictions,
        training_losses,
        out_of_sample_losses,
        training_indices,
        out_of_sample_indices,
        xmin,
        xmax,
        x,
        xtrain,
        ytrain,
        yvalid,
        rng,
    )

    # add new stumps to the forest:
    for _ in 1:Δntrees
        update!(args..., verbosity)
    end

    # make some noise, if allowed:
    verbosity ≥ 1 && @info "Trained $Δntrees additional stumps. "

    return StumpRegressorFitted(learner, args...)

end

# needed, because model is supervised:
LearnAPI.target(learner::StumpRegressor, observations) = last(observations)

LearnAPI.predict(model::StumpRegressorFitted, ::Point, x) =
    _predict(model.forest, x)

# accessor functions:
LearnAPI.learner(model::StumpRegressorFitted) = model.learner
LearnAPI.trees(model::StumpRegressorFitted) = model.forest
LearnAPI.predictions(model::StumpRegressorFitted) = model.predictions
LearnAPI.training_losses(model::StumpRegressorFitted) = model.training_losses
LearnAPI.out_of_sample_indices(model::StumpRegressorFitted) = model.out_of_sample_indices
LearnAPI.out_of_sample_losses(model::StumpRegressorFitted) = model.out_of_sample_losses
function LearnAPI.strip(model::StumpRegressorFitted)
    @unpack learner,
        forest,
        predictions,
        training_losses,
        out_of_sample_losses,
        training_indices,
        out_of_sample_indices,
        xmin,
        xmax,
        x,
        xtrain,
        ytrain,
        yvalid,
    rng = model

    return StumpRegressorFitted(
        learner,
        forest,
        Float64[],
        Float64[],
        Float64[],
        Int[],
        Int[],
        xmin,
        xmax,
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        rng,
    )
end

@trait(
    StumpRegressor,
    constructor = StumpRegressor,
    iteration_parameter = :ntrees,
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
        :(LearnAPI.trees),
        :(LearnAPI.predictions),
        :(LearnAPI.training_losses),
        :(LearnAPI.out_of_sample_indices),
        :(LearnAPI.out_of_sample_losses),
   )
)
