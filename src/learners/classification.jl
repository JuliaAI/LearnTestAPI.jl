# This file defines `ConstantClassifier()`

using LearnAPI
import LearnDataFrontEnds as FrontEnds
import MLCore
import CategoricalArrays
import CategoricalDistributions
import CategoricalDistributions.OrderedCollections.OrderedDict
import CategoricalDistributions.Distributions.StatsBase.proportionmap

# The implementation of a constant classifier below is not the simplest, but it
# demonstrates some patterns that apply more generally in classification, including
# inclusion of the canned data front end, `Sage`.

"""
    ConstantClassifier()

Instantiate a constant (dummy) classifier. Can predict `Point` or `Distribution` targets.

"""
struct ConstantClassifier end

struct ConstantClassifierFitted
    learner::ConstantClassifier
    probabilities
    names::Vector{Symbol}
    classes_seen
    codes_seen
    decoder
end

LearnAPI.learner(model::ConstantClassifierFitted) = model.learner

# add a data front end; `obs` will return objects with type `FrontEnds.Obs`:
const front_end = FrontEnds.Sage(code_type=:small)
LearnAPI.obs(learner::ConstantClassifier, data) =
    FrontEnds.fitobs(learner, data, front_end)
LearnAPI.obs(model::ConstantClassifierFitted, data) =
    obs(model, data, front_end)

# data deconstructors:
LearnAPI.features(learner::ConstantClassifier, data) =
    LearnAPI.features(learner, data, front_end)
LearnAPI.target(learner::ConstantClassifier, data) =
    LearnAPI.target(learner, data, front_end)

function LearnAPI.fit(
    learner::ConstantClassifier,
    observations::FrontEnds.Obs;
    verbosity=LearnAPI.default_verbosity(),
    )

    y = observations.target # integer "codes"
    names = observations.names
    classes_seen = observations.classes_seen
    codes_seen = sort(unique(y))
    decoder = observations.decoder

    d = proportionmap(y)
    # proportions ordered by key, i.e., by codes seen:
    probabilities = values(sort!(OrderedDict(d))) |> collect

    return ConstantClassifierFitted(
        learner,
        probabilities,
        names,
        classes_seen,
        codes_seen,
        decoder,
    )
end
LearnAPI.fit(learner::ConstantClassifier, data; kwargs...) =
    fit(learner, obs(learner, data); kwargs...)

function LearnAPI.predict(
    model::ConstantClassifierFitted,
    ::Point,
    observations::FrontEnds.Obs,
    )
    n = MLCore.numobs(observations)
    idx = argmax(model.probabilities)
    code_of_mode = model.codes_seen[idx]
    return model.decoder.(fill(code_of_mode, n))
end
LearnAPI.predict(model::ConstantClassifierFitted, ::Point, data) =
    predict(model, Point(), obs(model, data))

function LearnAPI.predict(
    model::ConstantClassifierFitted,
    ::Distribution,
    observations::FrontEnds.Obs,
    )
    n = MLCore.numobs(observations)
    probs = model.probabilities
    # repeat vertically to get rows of a matrix:
    probs_matrix = reshape(repeat(probs, n), (length(probs), n))'
    return CategoricalDistributions.UnivariateFinite(model.classes_seen, probs_matrix)
end
LearnAPI.predict(model::ConstantClassifierFitted, ::Distribution, data) =
        predict(model, Distribution(), obs(model, data))

# accessor function:
LearnAPI.feature_names(model::ConstantClassifierFitted) = model.names

@trait(
    ConstantClassifier,
    constructor = ConstantClassifier,
    kinds_of_proxy = (Point(),Distribution()),
    tags = ("classification",),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.clone),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.feature_names),
   )
)

true
