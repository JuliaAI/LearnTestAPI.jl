using LearnAPI
using Tables

# for testing:
using Test
using LearnTestAPI
import MLUtils
import DataFrames


# # TRANSFORMER TO SELECT SOME FEATURES (COLUMNS) OF A TABLE

# ## Implementation

# See later for a variation that stores the names of rejected features in the model
# object, for inspection by an accessor function.

struct Selector
    names::Vector{Symbol}
end
Selector(; names=Symbol[]) =  Selector(names) # LearnAPI.constructor defined later

# `fit` consumes no observational data, does no "learning", and just returns a thinly
# wrapped `learner` (to distinguish it from the learner in dispatch):
LearnAPI.fit(learner::Selector; verbosity=1) = Ref(learner)
LearnAPI.learner(model) = model[]

function LearnAPI.transform(model::Base.RefValue{Selector}, X)
    learner = LearnAPI.learner(model)
    table = Tables.columntable(X)
    names = Tables.columnnames(table)
    filtered_names = filter(in(learner.names), names)
    filtered_columns = (Tables.getcolumn(table, name) for name in filtered_names)
    filtered_table = NamedTuple{filtered_names}((filtered_columns...,))
    return Tables.materializer(X)(filtered_table)
end

# fit and transform in one go:
function LearnAPI.transform(learner::Selector, X)
    model = fit(learner)
    transform(model, X)
end

# note the necessity of overloading `is_static` (`fit` consumes no data):
@trait(
    Selector,
    constructor = Selector,
    tags = ("feature engineering",),
    is_static = true,
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.transform),
    ),
)

# ## Tests

learner = Selector(names=[:x, :w])
X = DataFrames.DataFrame(rand(3, 4), [:x, :y, :z, :w])
@testapi learner X verbosity=0

@testset "test a static transformer" begin
    model = fit(learner) # no data arguments!
    W = transform(model, X)
    @test W == DataFrames.DataFrame(Tables.matrix(X)[:,[1,4]], [:x, :w])
    @test W == transform(learner, X)
end


# # FEATURE SELECTOR THAT REPORTS BYPRODUCTS OF SELECTION PROCESS

# ## Implememtation

# This a variation of `Selector` above that stores the names of rejected features in the
# output of `fit`, for inspection by an accessor function called `rejected`.

struct FancySelector
    names::Vector{Symbol}
end

"""
    FancySelector(; names=Symbol[])

Instantiate a feature selector that exposes the names of rejected features. Inputs for
transform are expected to be tables.

```julia
learner = FancySelector(names=[:x, :w])
X = DataFrames.DataFrame(rand(3, 4), [:x, :y, :z, :w])
model = fit(learner) # no data arguments!
transform(model, X)  # mutates `model`
@assert rejected(model) == [:y, :z]
```

"""
FancySelector(; names=Symbol[]) =  FancySelector(names)

mutable struct FancySelectorFitted
    learner::FancySelector
    rejected::Vector{Symbol}
    FancySelectorFitted(learner) = new(learner)
end
LearnAPI.learner(model::FancySelectorFitted) = model.learner
rejected(model::FancySelectorFitted) = model.rejected

# Here we are wrapping `learner` with a place-holder for the `rejected` feature names.
LearnAPI.fit(learner::FancySelector; verbosity=1) = FancySelectorFitted(learner)

# output the filtered table and add `rejected` field to model (mutatated!)
function LearnAPI.transform(model::FancySelectorFitted, X)
    table = Tables.columntable(X)
    names = Tables.columnnames(table)
    keep = LearnAPI.learner(model).names
    filtered_names = filter(in(keep), names)
    model.rejected = setdiff(names, filtered_names)
    filtered_columns = (Tables.getcolumn(table, name) for name in filtered_names)
    filtered_table = NamedTuple{filtered_names}((filtered_columns...,))
    return Tables.materializer(X)(filtered_table)
end

# fit and transform in one step:
function LearnAPI.transform(learner::FancySelector, X)
    model = fit(learner)
    transform(model, X)
end

# note the necessity of overloading `is_static` (`fit` consumes no data):
@trait(
    FancySelector,
    constructor = FancySelector,
    is_static = true,
    tags = ("feature engineering",),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.transform),
        :(MyPkg.rejected), # accessor function not owned by LearnAPI.jl,
    )
)

# ## Tests

learner = FancySelector(names=[:x, :w])
X = DataFrames.DataFrame(rand(3, 4), [:x, :y, :z, :w])
@testapi learner X verbosity=0

@testset "test a variation that reports byproducts" begin
    model = fit(learner) # no data arguments!
    @test !isdefined(model, :reject)
    filtered =  DataFrames.DataFrame(Tables.matrix(X)[:,[1,4]], [:x, :w])
    @test transform(model, X) == filtered
    @test transform(learner, X) == filtered
    @test rejected(model) == [:y, :z]
end

true
