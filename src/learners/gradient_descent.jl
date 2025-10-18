# THIS FILE IS NOT INCLUDED BUT EXISTS AS AN IMPLEMENTATION EXEMPLAR

# TODO: This file should be updated after release of CategoricalDistributions 0.2 as
# `classes` is deprecated there.

# This file defines:
# - `PerceptronClassifier(; epochs=50, optimiser=Optimisers.Adam(), rng=Random.default_rng())

using LearnAPI
using Random
using Statistics
using StableRNGs
import Optimisers
import Zygote
import NNlib
import CategoricalDistributions
import CategoricalDistributions: pdf, mode
import ComponentArrays

# We implement a simple perceptron classifier to illustrate some common patterns for
# gradient descent algorithms. This includes implementation of the following methods:

# - `update`
# - `update_observations`
# - `iteration_parameter`
# - `training_losses`
# - `obs` for pre-processing (non-tabular) classification training data
# - `predict(learner, ::Distribution, Xnew)`

# For simplicity, we use single-observation batches for gradient descent updates, and we
# may dodge some optimizations.

# This is an example of a probability-predicting classifier.


# ## Helpers

"""
    brier_loss(probs, hot)

Return Brier (quadratic) loss.

- `probs`: predicted probability vector
- `hot`: corresponding ground truth observation, as a one-hot encoded `BitVector`

"""
function brier_loss(probs, hot)
    offset = 1 + sum(probs.^2)
    return offset - 2*(sum(probs.*hot))
end

"""
    corefit(perceptron, optimiser, X, y_hot, epochs, state, verbosity)

Return updated `perceptron`, `state`, and training losses by carrying out gradient descent
for the specified number of `epochs`.

- `perceptron`: component array with components `weights` and `bias`
- `optimiser`: optimiser from Optimiser.jl
- `X`: feature matrix, of size `(p, n)`
- `y_hot`: one-hot encoded target, of size `(nclasses, n)`
- `epochs`: number of epochs
- `state`: optimiser state

"""
function corefit(perceptron, X, y_hot, epochs, state, verbosity)
    n = size(y_hot) |> last
    losses = map(1:epochs) do _
        total_loss = zero(Float32)
        for i in 1:n
            loss, grad = Zygote.withgradient(perceptron) do p
                probs = p.weights*X[:,i] + p.bias |> NNlib.softmax
                brier_loss(probs, y_hot[:,i])
            end
            ∇loss = only(grad)
            state, perceptron = Optimisers.update(state, perceptron, ∇loss)
            total_loss += loss
        end
        # make some noise, if allowed:
        verbosity > 0 && @info "Training loss: $total_loss"
        total_loss
    end
    return perceptron, state, losses
end


# ## Implementation

# ### Learner

# no docstring here - that goes with the constructor;
# SOME FIELDS LEFT ABSTRACT FOR SIMPLICITY
struct PerceptronClassifier
    epochs::Int
    optimiser # an optmiser from Optimsers.jl
    rng
end

"""
    PerceptronClassifier(; epochs=50, optimiser=Optimisers.Adam(), rng=Random.default_rng())

Instantiate a perceptron classifier.

Train an instance, `learner`, by doing `model = fit(learner, X, y)`, where

-  `X is a `Float32` matrix, with observations-as-columns
-  `y` (target) is some one-dimensional `CategoricalArray`.

Get probabilistic predictions with `predict(model, Xnew)` and
point predictions with `predict(model, Point(), Xnew)`.

# Warm restart options

    update(model, newdata, :epochs=>n, other_replacements...)

If `Δepochs = n - perceptron.epochs` is non-negative, then return an updated model, with
the weights and bias of the previously learned perceptron used as the starting state in
new gradient descent updates for `Δepochs` epochs, and using the provided `newdata`
instead of the previous training data. Any other hyperparaameter `replacements` are also
adopted. If `Δepochs` is negative or not specified, instead return `fit(learner,
newdata)`, where `learner=LearnAPI.clone(learner; epochs=n, replacements....)`.

    update_observations(model, newdata, replacements...)

Return an updated model, with the weights and bias of the previously learned perceptron
used as the starting state in new gradient descent updates. Adopt any specified
hyperparameter `replacements` (properties of `LearnAPI.learner(model)`).

"""
PerceptronClassifier(; epochs=50, optimiser=Optimisers.Adam(), rng=Random.default_rng()) =
    PerceptronClassifier(epochs, optimiser, rng)


# Type for internal representation of data (output of `obs(learner, data)`):
struct PerceptronClassifierObs
    X::Matrix{Float32}
    y_hot::BitMatrix  # one-hot encoded target
    classes           # the (ordered) pool of `y`, as `CategoricalValue`s
end

# For pre-processing the training data:
function LearnAPI.obs(::PerceptronClassifier, data::Tuple)
    X, y = data
    classes = CategoricalDistributions.classes(y)
    y_hot = classes .== permutedims(y) # one-hot encoding
    return PerceptronClassifierObs(X, y_hot, classes)
end
LearnAPI.obs(::PerceptronClassifier, observations::PerceptronClassifierObs) =
    observations # involutivity

# helper:
function decode(y_hot, classes)
    n = size(y_hot, 2)
    [only(classes[y_hot[:,i]]) for i in 1:n]
end

# implement `RadomAccess()` interface for output of `obs`:
Base.length(observations::PerceptronClassifierObs) = size(observations.y_hot, 2)
Base.getindex(observations::PerceptronClassifierObs, I) = PerceptronClassifierObs(
    observations.X[:, I],
    observations.y_hot[:, I],
    observations.classes,
)

# training data deconstructors:
LearnAPI.target(
    learner::PerceptronClassifier,
    observations::PerceptronClassifierObs,
) = decode(observations.y_hot, observations.classes)
LearnAPI.target(learner::PerceptronClassifier, data) =
    LearnAPI.target(learner, obs(learner, data))
LearnAPI.features(
    learner::PerceptronClassifier,
    observations::PerceptronClassifierObs,
) = observations.X
LearnAPI.features(learner::PerceptronClassifier, data) =
    LearnAPI.features(learner, obs(learner, data))

# Note that data consumed by `predict` needs no pre-processing, so no need to overload
# `obs(model, data)`.


# ### Fitting and updating

# For wrapping outcomes of learning:
struct PerceptronClassifierFitted
    learner::PerceptronClassifier
    perceptron  # component array storing weights and bias
    state       # optimiser state
    classes     # target classes
    losses
end

LearnAPI.learner(model::PerceptronClassifierFitted) = model.learner

# `fit` for pre-processed data (output of `obs(learner, data)`):
function LearnAPI.fit(
    learner::PerceptronClassifier,
    observations::PerceptronClassifierObs;
    verbosity=LearnAPI.default_verbosity(),
    )

    # unpack hyperparameters:
    epochs = learner.epochs
    optimiser = learner.optimiser
    rng = deepcopy(learner.rng) # to prevent mutation of `learner`!

    # unpack data:
    X = observations.X
    y_hot = observations.y_hot
    classes = observations.classes
    nclasses = length(classes)

    # initialize bias and weights:
    weights = randn(rng, Float32, nclasses, p)
    bias = zeros(Float32, nclasses)
    perceptron = (; weights, bias) |> ComponentArrays.ComponentArray

    # initialize optimiser:
    state = Optimisers.setup(optimiser, perceptron)

    perceptron, state, losses = corefit(perceptron, X, y_hot, epochs, state, verbosity)

    return PerceptronClassifierFitted(learner, perceptron, state, classes, losses)
end

# `fit` for unprocessed data:
LearnAPI.fit(learner::PerceptronClassifier, data; kwargs...) =
    fit(learner, obs(learner, data); kwargs...)

# see the `PerceptronClassifier` docstring for `update_observations` logic.
function LearnAPI.update_observations(
    model::PerceptronClassifierFitted,
    observations_new::PerceptronClassifierObs,
    replacements...;
    verbosity=LearnAPI.default_verbosity(),
    )

    # unpack data:
    X = observations_new.X
    y_hot = observations_new.y_hot
    classes = observations_new.classes
    nclasses = length(classes)

    classes == model.classes || error("New training target has incompatible classes.")

    learner_old = LearnAPI.learner(model)
    learner = LearnAPI.clone(learner_old, replacements...)

    perceptron = model.perceptron
    state = model.state
    losses = model.losses
    epochs = learner.epochs

    perceptron, state, losses_new = corefit(perceptron, X, y_hot, epochs, state, verbosity)
    losses = vcat(losses, losses_new)

    return PerceptronClassifierFitted(learner, perceptron, state, classes, losses)
end
LearnAPI.update_observations(model::PerceptronClassifierFitted, data, args...; kwargs...) =
    update_observations(model, obs(LearnAPI.learner(model), data), args...; kwargs...)

# see the `PerceptronClassifier` docstring for `update` logic.
function LearnAPI.update(
    model::PerceptronClassifierFitted,
    observations::PerceptronClassifierObs,
    replacements...;
    verbosity=LearnAPI.default_verbosity(),
    )

    # unpack data:
    X = observations.X
    y_hot = observations.y_hot
    classes = observations.classes
    nclasses = length(classes)

    classes == model.classes || error("New training target has incompatible classes.")

    learner_old = LearnAPI.learner(model)
    learner = LearnAPI.clone(learner_old, replacements...)
    :epochs in keys(replacements) || return fit(learner, observations; verbosity)

    perceptron = model.perceptron
    state = model.state
    losses = model.losses

    epochs = learner.epochs
    Δepochs = epochs - learner_old.epochs
    epochs < 0 && return fit(model, learner; verbosity)

    perceptron, state, losses_new =
        corefit(perceptron, X, y_hot, Δepochs, state, verbosity)
    losses = vcat(losses, losses_new)

    return PerceptronClassifierFitted(learner, perceptron, state, classes, losses)
end
LearnAPI.update(model::PerceptronClassifierFitted, data, args...; kwargs...) =
    update(model, obs(LearnAPI.learner(model), data), args...; kwargs...)


# ### Predict

function LearnAPI.predict(model::PerceptronClassifierFitted, ::Distribution, Xnew)
    perceptron = model.perceptron
    classes = model.classes
    probs = perceptron.weights*Xnew .+ perceptron.bias |> NNlib.softmax
    return CategoricalDistributions.UnivariateFinite(classes, probs')
end

LearnAPI.predict(model::PerceptronClassifierFitted, ::Point, Xnew) =
    mode.(predict(model, Distribution(), Xnew))


# ### Accessor functions

LearnAPI.training_losses(model::PerceptronClassifierFitted) = model.losses


# ### Traits

@trait(
    PerceptronClassifier,
    constructor = PerceptronClassifier,
    iteration_parameter = :epochs,
    kinds_of_proxy = (Distribution(), Point()),
    tags = ("classification", "iterative algorithms", "incremental algorithms"),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.clone),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.update),
        :(LearnAPI.update_observations),
        :(LearnAPI.predict),
        :(LearnAPI.training_losses),
   )
)
