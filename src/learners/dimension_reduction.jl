# This file defines `TruncatedSVD(; codim=1)`

using LearnAPI
using LinearAlgebra
import LearnDataFrontEnds as FrontEnds


# # DIMENSION REDUCTION USING TRUNCATED SVD DECOMPOSITION

# Recall that truncated SVD reduction is the same as PCA reduction, but without
# centering.

# Some struct fields are left abstract for simplicity.

# ## Implementation

struct TruncatedSVD
    codim::Int
end

"""
    TruncatedSVD(; codim=1)

Instantiate a truncated singular value decomposition algorithm for reducing the dimension
of observations by `codim`.

Data can be provided to `fit` or `transform` in any form supported by the `Tarragon` data
front end at LearnDataFrontEnds.jl. However, the outputs of `transform` and
`inverse_transform` are always matrices.


```julia
learner = Truncated()
X = rand(3, 100)  # 100 observations in 3-space
model = fit(learner, X)
W = transform(model, X)
X_reconstructed = inverse_transform(model, W)
LearnAPI.extras(model) # returns indim, outdim and singular values
```

The following fits and transforms in one go:

```julia
W = transform(learner, X)
```
"""
TruncatedSVD(; codim=1) = TruncatedSVD(codim)

struct TruncatedSVDFitted
    learner::TruncatedSVD
    U  # of size `(p - codim, p)` for input observations in `p`-space
    Ut # of size `(p, p - codim)`
    singular_values
end

LearnAPI.learner(model::TruncatedSVDFitted) = model.learner

# add a canned data front end; `obs` will return objects of type `FrontEnds.Obs`:
LearnAPI.obs(learner::TruncatedSVD, data) =
    FrontEnds.fitobs(learner, data, FrontEnds.Tarragon())
LearnAPI.obs(model::TruncatedSVDFitted, data) =
    obs(model, data, FrontEnds.Tarragon())

# training data deconstructor:
LearnAPI.features(learner::TruncatedSVD, data) =
    LearnAPI.features(learner, data, FrontEnds.Tarragon())

function LearnAPI.fit(
    learner::TruncatedSVD,
    observations::FrontEnds.Obs;
    verbosity=LearnAPI.default_verbosity(),
    )

    # unpack hyperparameters:
    codim = learner.codim
    X = observations.features
    p, n = size(X)
    n ≥ p || error("Insufficient number observations. ")
    outdim = p - codim

    # apply core algorithm:
    result = svd(X)
    Ut = adjoint(@view result.U[:,1:outdim])
    U = adjoint(Matrix(Ut))
    singular_values = result.S
    dropped = singular_values[end - codim:end]

    verbosity > 0 &&
        @info "Singular values for dropped components: $dropped"

    return TruncatedSVDFitted(learner, U, Ut, singular_values)

end
LearnAPI.fit(learner::TruncatedSVD, data; kwargs...) =
    LearnAPI.fit(learner, LearnAPI.obs(learner, data); kwargs...)

LearnAPI.transform(model::TruncatedSVDFitted, observations::FrontEnds.Obs) =
    model.Ut*(observations.features)
LearnAPI.transform(model::TruncatedSVDFitted, data) =
    LearnAPI.transform(model, obs(model, data))

# convenience fit-transform:
LearnAPI.transform(learner::TruncatedSVD, data; kwargs...) =
    transform(fit(learner, data; kwargs...), data)

LearnAPI.inverse_transform(model::TruncatedSVDFitted, W::AbstractMatrix) = model.U*W

# accessor function:
function LearnAPI.extras(model::TruncatedSVDFitted)
    outdim, indim = size(model.Ut)
    singular_values = model.singular_values
    return (; outdim, indim, singular_values)
end

# for illustration, we drop singular values in strip:
LearnAPI.strip(model::TruncatedSVDFitted) =
    TruncatedSVDFitted(model.learner, model.U, model.Ut, nothing)

@trait(
    TruncatedSVD,
    constructor = TruncatedSVD,
    tags = ("dimension reduction", "transformers"),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.clone),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.transform),
        :(LearnAPI.inverse_transform),
        :(LearnAPI.extras),
    ),
    load_path  = "LearnTestAPI.TruncatedSVD",
    pkg_name = "LearnTestAPI",
    pkg_license = "MIT expat",
    doc_url = "https://juliaai.github.io/LearnAPI.jl/dev/testing_an_implementation/",
)
