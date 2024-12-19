# This file defines:

# - `Ridge(; lambda=0.1)`
# - `BabyRidge(; lambda=0.1)`

using LearnAPI
using Tables
using LinearAlgebra

# # NAIVE RIDGE REGRESSION WITH NO INTERCEPTS

# We overload `obs` to expose internal representation of data. See later for a simpler
# variation using the `obs` fallback.

struct Ridge
    lambda::Float64
end

"""
    Ridge(; lambda=0.1)

Instantiate a ridge regression learner, with regularization of `lambda`.

"""
Ridge(; lambda=0.1) = Ridge(lambda) # LearnAPI.constructor defined later

struct RidgeFitObs{T,M<:AbstractArray{T}}
    A::M  # p x n
    names::Vector{Symbol}
    y::Vector{T}
end

struct RidgeFitted{T,F}
    learner::Ridge
    coefficients::Vector{T}
    feature_importances::F
    names::Vector{Symbol}
end

LearnAPI.learner(model::RidgeFitted) = model.learner

Base.getindex(data::RidgeFitObs, I) =
    RidgeFitObs(data.A[:,I], data.names, data.y[I])
Base.length(data::RidgeFitObs) = length(data.y)

# observations for consumption by `fit`:
function LearnAPI.obs(::Ridge, data)
    X, y = data
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    RidgeFitObs(Tables.matrix(table)', names, y)
end
LearnAPI.obs(::Ridge, data::RidgeFitObs) = data

# for observations:
function LearnAPI.fit(learner::Ridge, observations::RidgeFitObs; verbosity=1)

    # unpack hyperparameters and data:
    lambda = learner.lambda
    A = observations.A
    names = observations.names
    y = observations.y

    # apply core learner:
    coefficients = (A*A' + learner.lambda*I)\(A*y) # 1 x p matrix

    # determine crude feature importances:
    feature_importances =
        [names[j] => abs(coefficients[j]) for j in eachindex(names)]
    sort!(feature_importances, by=last) |> reverse!

    # make some noise, if allowed:
    verbosity > 0 &&
        @info "Features in order of importance: $(first.(feature_importances))"

    return RidgeFitted(learner, coefficients, feature_importances, names)

end

# for unprocessed `data = (X, y)`:
LearnAPI.fit(learner::Ridge, data; kwargs...) =
    fit(learner, obs(learner, data); kwargs...)

# extracting stuff from training data:
LearnAPI.target(::Ridge, data) = last(data)
LearnAPI.target(::Ridge, observations::RidgeFitObs) = observations.y
LearnAPI.features(::Ridge, observations::RidgeFitObs) = observations.A

# observations for consumption by `predict`:
LearnAPI.obs(::RidgeFitted, X) = Tables.matrix(X)'
LearnAPI.obs(::RidgeFitted, X::AbstractMatrix) = X

# matrix input:
LearnAPI.predict(model::RidgeFitted, ::Point, observations::AbstractMatrix) =
        observations'*model.coefficients

# tabular input:
LearnAPI.predict(model::RidgeFitted, ::Point, Xnew) =
        predict(model, Point(), obs(model, Xnew))

# accessor function:
LearnAPI.feature_importances(model::RidgeFitted) = model.feature_importances
LearnAPI.feature_names(model::RidgeFitted) = model.names

LearnAPI.strip(model::RidgeFitted) =
    RidgeFitted(model.learner, model.coefficients, nothing, Symbol[])

@trait(
    Ridge,
    constructor = Ridge,
    human_name = "ridge regression",
    kinds_of_proxy = (Point(),),
    tags = ("regression",),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.feature_names),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.feature_importances),
   )
)

# convenience method:
LearnAPI.fit(learner::Ridge, X, y; kwargs...) =
    fit(learner, (X, y); kwargs...)


# # VARIATION OF RIDGE REGRESSION THAT USES FALLBACK OF LearnAPI.obs

# no docstring here - that goes with the constructor
struct BabyRidge
    lambda::Float64
end

"""
    BabyRidge(; lambda=0.1)

Instantiate a ridge regression learner, with regularization of `lambda`.

"""
BabyRidge(; lambda=0.1) = BabyRidge(lambda) # LearnAPI.constructor defined later

struct BabyRidgeFitted{T,F}
    learner::BabyRidge
    coefficients::Vector{T}
    feature_importances::F
end

function LearnAPI.fit(learner::BabyRidge, data; verbosity=1)

    X, y = data

    lambda = learner.lambda
    table = Tables.columntable(X)
    names = Tables.columnnames(table) |> collect
    A = Tables.matrix(table)'

    # apply core learner:
    coefficients = (A*A' + learner.lambda*I)\(A*y) # vector

    feature_importances = nothing
#    feature_importances[1:2]
    return BabyRidgeFitted(learner, coefficients, feature_importances)

end

# extracting stuff from training data:
LearnAPI.target(::BabyRidge, data) = last(data)

LearnAPI.learner(model::BabyRidgeFitted) = model.learner

LearnAPI.predict(model::BabyRidgeFitted, ::Point, Xnew) =
    Tables.matrix(Xnew)*model.coefficients

LearnAPI.strip(model::BabyRidgeFitted) =
    BabyRidgeFitted(model.learner, model.coefficients, nothing)

@trait(
    BabyRidge,
    constructor = BabyRidge,
    kinds_of_proxy = (Point(),),
    tags = ("regression",),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
   )
)

# convenience method:
LearnAPI.fit(learner::BabyRidge, X, y; kwargs...) =
    fit(learner, (X, y); kwargs...)
