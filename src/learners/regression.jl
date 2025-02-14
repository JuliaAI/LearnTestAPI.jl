# This file defines:

# - `Ridge(; lambda=0.1)`
# - `BabyRidge(; lambda=0.1)`

using LearnAPI
using Tables
using LinearAlgebra
import LearnDataFrontEnds as FrontEnds

# # NAIVE RIDGE REGRESSION WITH NO INTERCEPTS

# We implement a canned data front end. See `BabyRidgeRegressor` below for a no-frills
# version.

# no docstring here; that goes with the constructor
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

# add a canned data front end; `obs` will return objects of type `FrontEnds.Obs`:
const frontend = FrontEnds.Saffron(view=true)
LearnAPI.obs(learner::Ridge, data) = FrontEnds.fitobs(learner, data, frontend)
LearnAPI.obs(model::RidgeFitted, data) = obs(model, data, frontend)

# training data deconstructors:
LearnAPI.features(learner::Ridge, data) = LearnAPI.features(learner, data, frontend)
LearnAPI.target(learner::Ridge, data) = LearnAPI.target(learner, data, frontend)

function LearnAPI.fit(learner::Ridge, observations::FrontEnds.Obs; verbosity=1)

    # unpack hyperparameters and data:
    lambda = learner.lambda
    A = observations.features
    names = observations.names
    y = observations.target

    # apply core learner:
    coefficients = (A*A' + learner.lambda*I)\(A*y) #  p x 1 matrix

    # determine crude feature importances:
    feature_importances =
        [names[j] => abs(coefficients[j]) for j in eachindex(names)]
    sort!(feature_importances, by=last) |> reverse!

    # make some noise, if allowed:
    verbosity > 0 &&
        @info "Features in order of importance: $(first.(feature_importances))"

    return RidgeFitted(learner, coefficients, feature_importances, names)

end
LearnAPI.fit(learner::Ridge, data; kwargs...) =
    fit(learner, obs(learner, data); kwargs...)

LearnAPI.predict(model::RidgeFitted, ::Point, observations::FrontEnds.Obs) =
    (observations.features)'*model.coefficients
LearnAPI.predict(model::RidgeFitted, kind_of_proxy, data) =
    LearnAPI.predict(model, kind_of_proxy, obs(model, data))

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
        :(LearnAPI.clone),
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


# # VARIATION OF RIDGE REGRESSION WITHOUT DATA FRONT END

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
        :(LearnAPI.clone),
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
