"""
    LearnTestAPI

Module for testing implementations of the interfacde defined in
[LearnAPI.jl](https://juliaai.github.io/LearnAPI.jl/dev/).

If your package defines an object `learner` implementing the interface, then put something
like this in your test suite:

```julia
using LearnTestAPI

# create some test data:
X = ...
y = ...
data = (X, y)

# bump verbosity to debug:
@testapi learner data verbosity=1
```

Once tests pass, set `verbosity=0` to suppress the detailed logging.

For details and options see [`LearnTestAPI.@testapi`](@ref)

"""
module LearnTestAPI

using LearnAPI
import Test
import Serialization
import MLUtils
import StableRNGs
import InteractiveUtils
import MacroTools

include("tools.jl")


# # LOGGING

ERR_UNSUPPORTED_KWARG(arg) = ArgumentError(
    "Got unsupported keyword argument `$arg`. "
)

const LOUD = "- specify `verbosity=0` for silent testing. ------"

const QUIET = "- specify `verbosity=1` if debugging"

const CONSTRUCTOR = """

    Testing that learner can be reconstructed from its constructors.
    [Reference](https://juliaai.github.io/LearnAPI.jl/dev/reference/#learners).

  """
const IS_COMPOSITE = """

    `LearnAPI.is_composite(learner)` returns `false`. Testing no properties of `learner`
    appear to be other
    learners. [Reference](https://juliaai.github.io/LearnAPI.jl/dev/reference/#learners).

  """
const WARN_DOCUMENTATION = """

    The learner constructor is undocumented.
    [Reference](https://juliaai.github.io/LearnAPI.jl/dev/reference/#Documentation).

  """
const OBLIGATORY_FUNCTIONS =
    join(map(ex->"`:$ex`", LearnAPI.functions()[1:4]), ", ", " and ")
const FUNCTIONS = """

    Testing that `LearnAPI.functions(learner)` includes $OBLIGATORY_FUNCTIONS.

  """
const FUNCTIONS3 = """

    Testing that `LearnAPI.functions(learner)` includes `:(LearnAPI.features).`

  """
const TAGS = """

    Testing that `LearnAPI.tags(learner)` has correct form. List allowed tags with
    `LearnAPII.tags()`.

  """
const KINDS_OF_PROXY = """

    Testing that `LearnAPI.kinds_of_proxy(learner)` is overloaded and has valid form.
    [Reference](https://juliaai.github.io/LearnAPI.jl/dev/kinds_of_target_proxy/). List
    allowed kinds of proxy with `LearnAPI.kinds_of_proxy()`.

  """
const FIT_IS_STATIC = """

    `LearnAPI.is_static(learner)` is `true`. Therefore attempting to call
    `fit(learner)`.

  """
const FIT_IS_NOT_STATIC = """

    Attempting to call `fit(learner, data)`. If you implemented `fit(learner)` instead,
    then you need to arrange `LearnAPI.is_static(learner) == true`.

  """
const LEARNER = """

    Attempting to call `LearnAPI.strip(model)` and to check that applying
    `LearnAPI.learner` to the result or to `model` returns `learner` in both cases.

  """
const FUNCTIONS2 = """

    Checking for overloaded functions missing from `LearnAPI.functions(learner)`.  Run
    `??LearnAPI.functions` for a checklist.

  """
const FEATURES = """

    Attempting to call `LearnAPI.features(learner, data)`.

  """
const OBS_INVOLUTIVITY = """

    Testing that `obs(model, obs(model, data)) == obs(model, data)` (involutivity).

  """

const SERIALIZATION = """

    Checking that `LearnAPI.strip(model)` can be Julia-serialized/deserialized.

  """
const PREDICT_HAS_NO_FEATURES = """

    Attempting to call `predict(model, kind)` for each `kind` in
    `LearnAPI.kinds_of_proxy(learner)`.

  """
const PREDICT_HAS_FEATURES = """

    Since `X = LearnAPI.features(learner, data)` is not `nothing`, we are attempting to
    call `predict(model, kind, X)` for each `kind` in `LearnAPI.kinds_of_proxy(learner)`.

  """
const STRIP = """

    Checking that `predict(model, ...)` gives the same answer if we replace `model` with
    `LearnAPI.strip(model)`.

  """
const STRIP2 = """

    Checking that `predict(model, X)` can be called on the deserialized version of `model`
    without a change in return value.

  """
const OBS_AND_PREDICT = """

    Testing that replacing `X` in `predict(learner, kind, X)` with `obs(model, X)` gives
    the same output, for each supported `kind` of proxy.

  """
const TRANSFORM_HAS_NO_FEATURES = """

    Attempting to call `transform(model)`.

  """
const TRANSFORM_HAS_FEATURES = """

    Since `X = LearnAPI.features(learner, data)` is not `nothing`, we are attempting to
    call `transform(model, X)`.

  """
const STRIP_TRANSFORM = """

    Testing that `transform(model, X)` gives the same answer if we replace `model` with
    `LearnAPI.strip(model)`.

  """
const STRIP2_TRANSFORM = """

    Testing  that `transform(model, X)` can be called using the deserialized version of
    `model` without a change in return value.

  """
const OBS_AND_TRANSFORM = """

    Testing that replacing `X` in `transform(learner, X)` with `obs(model, X)` gives the
    same output.

  """
const INVERSE_TRANSFORM = """

    Testing that `inverse_transform(model, W)` executes, where `W` is the output of
    `transform(model, X)` and `X = LearnAPI.features(learner, data)`.

  """
const STRIP_INVERSE = """

    Testing that `inverse_transform(model, X)` gives the same answer if we replace
    `model` with `LearnAPI.strip(model)`.

  """
const STRIP2_INVERSE = """

    Testing that `inverse_transform(model, X)` can be called using the deserialized
    version of `model` without a change in return value.

  """
const OBS_INVOLUTIVITY_FIT =
    """

    Testing that `obs(learner, obs(learner, X)) == obs(learner, X)` (involutivity).

  """
const OBS_AND_FIT = """

    Testing that replacing `data` with `obs(learner, data)` in the call `fit(learner,
    data)` makes no difference to `predict`, `transform` or `inverse_transform` outcomes
    (where implemented).

  """
const SELECTED_FOR_FIT = """

    Testing that we can select all observations from `_data = LearnAPI.obs(learner,
    data)`, using the interface declared by `LearnAPI.data_interface(learner)`. For
    example, if this interface is `LearnAPI.RandomAccess()`, the attempted selection is
    `data3 = MLUtils.getobs(_data, 1:MLUtils.numobs(_data))`. For this interface, also
    testing that we can call `fit` on the selection, to obtain new `fit` result
    `model3`.

  """
const SELECTED = """

    Testing that we can select all observations from `_X = LearnAPI.obs(model, X)`,
    using the interface declared by `LearnAPI.data_interface(learner)`. For example, if
    this interface is `LearnAPI.RandomAccess()`, the attempted selection is
    `X3 = MLUtils.getobs(_X, 1:MLUtils.numobs(_X))`.

  """
const PREDICT_ON_SELECTIONS1 = """

    Testing that `predict(model, X)` is unchanged by replacing `X` with `X3`
    ("subsampled" features). If not described in the log, re-run `@testapi` with
    `verbosity=1` for further explanation of `X3`.

  """
const TRANSFORM_ON_SELECTIONS1 = """

    Testing that `transform(model, X)` is unchanged by replacing `X` with `X3`
    ("subsampled" features). If not described in the log, re-run `@testapi` with
    `verbosity=1` for further  explanation of `X3`.

  """
const PREDICT_ON_SELECTIONS2 = """

    Testing that `predict(model, X)` is unchanged by replacing `model` with `model3`
    (trained on "subsampled" `fit` data) and/or replacing `X` with `X3`
    ("subsampled" features). If not described in the log, re-run `@testapi` with
    `verbosity=1` for further explanation of `model3` and `X3`.

  """
const TRANSFORM_ON_SELECTIONS2 = """

    Testing that `transform(model, X)` is unchanged by replacing `model` with `model3`
    (trained on "subsampled" `fit` data) and/or replacing `X` with `X3`
    ("subsampled" features). If not described in the log, re-run `@testapi` with
    `verbosity=1` for further explanation of `model3` and `X3`.

  """

const KINDS_OF_PROXY2 = """

    `LearnAPI.predict` is apparently overloaded. Checking `LearnAPI.kinds_of_proxy

  """


# # METAPROGRAMMING HELPERS

"""
    LearaAPI.verb(ex)

*Private method.*

If `ex` is a specification of `verbosity`, such as `:(verbosity=1)`, then return the
specified value; otherwise, return `nothing`.

"""
verb(ex) = nothing
function verb(ex::Expr)
    if ex.head == :(=)
        ex.args[1] == :verbosity || throw(ERR_UNSUPPORTED_KWARG(ex.args[1]))
        return ex.args[2] # the actual verbosity value
    end
    return nothing
end

"""
    LearnAPI.filter_out_verbosity(exs)

*Private method.*

Return `(filtered_exs, verbosity)` where `filtered_exs` is `exs` with any `verbosity`
specification dropped, and `verbosity` is the verbosity value (`1` if not
specified).

"""
function filter_out_verbosity(exs)
    verb = 1
    exs = filter(exs) do ex
        v = LearnTestAPI.verb(ex)
        keep = isnothing(v)
        keep || (verb = v)
        keep
    end
    return exs, verb
end


# # THE MAIN TEST MACRO

"""
    @testapi learner dataset1, dataset2 ... verbosity=1

Test that `learner` correctly implements the LearnAPI.jl interface, by checking
contracts against one or more data sets.

```julia
using LearnTestAPI

X = (
    feautre1 = [1, 2, 3],
    feature2 = ["a", "b", "c"],
    feature3 = [10.0, 20.0, 30.0],
)

@testapi MyFeatureSelector(; features=[:feature3,]) X verbosity=1
```

# Extended help

# Assumptions

In some tests strict `==` is enforced on the output of `predict` or `transform`, unless
`isapprox` is also implemented. If `predict` outputs categorical vectors, for example,
then requiring `==` in a test is appropriate. On the other hand, if `predict` outputs some
abstract vector of eltype `Float32`, it will be necessary that `isapprox` is implemented
for that vector type, because the strict test `==` is likely to fail. These comments apply
to more complicated objects, such as probability distributions or sampleable objects: If
`==` is likely to fail in "benign" cases, be sure `isapprox` is implemented. See
[`LearnTestAPI.isnear`](@ref) for the exact test applied.

# What is not tested?

When `verbosity=1` (the default) the test log describes all contracts tested.

The following are *not* tested:

- That the output of `LearnAPI.target(learner, data)` is indeed a target, in the sense
  that it can be paired, in some way, with the output of `predict`. Such a test would be
  to suitably pair the output with a predicted proxy for the target, using, say, a proper
  scoring rule, in the case of probabilistic predictions.

- That `inverse_transform` is an approximate left or right inverse to `transform`

- That the one-line convenience methods, `transform(learner, ...)` or `predict(learner,
  ...)`, where implemented, have the same effect as the two-line calls they combine.

Whenever the internal `learner` algorithm involves case distinctions around data or
hyperparameters, it is recommended that multiple datasets, and learners with a variety of
hyperparameter settings, are explicitly tested.

# Role of datasets in tests

Each `dataset` is used as follows.

If `LearnAPI.is_static(learner) == false`, then:

- `dataset` is passed to `fit` and, if necessary, its `update` cousins

- If `X = LearnAPI.features(learner, dataset) == nothing`, then `predict` and/or
  `transform` are called with no data. Otherwise, they are called with `X`.

If instead `LearnAPI.is_static(learner) == true`, then `fit` and its cousins are called
without any data, and `dataset` is passed directly to `fit` and/or `transform`.

"""
macro testapi(learner, data...)

    data, verbosity = filter_out_verbosity(data)

    quote
        import Test
        import Serialization
        import MLUtils
        import LearnAPI
        import InteractiveUtils

        learner = $(esc(learner))
        verbosity=$verbosity
        _human_name = LearnAPI.human_name(learner)
        _data_interface = LearnAPI.data_interface(learner)
        _is_static = LearnAPI.is_static(learner)

        if isnothing(verbosity) || verbosity > 0
            @info "------ @testapi - $_human_name "*$LOUD
        else
            verbosity > -1 && @info "@testapi - $_human_name "*$QUIET
        end

        LearnTestAPI.@logged_testset $CONSTRUCTOR verbosity begin
            Test.@test LearnAPI.clone(learner) == learner
        end

        if !LearnAPI.is_composite(learner)
            LearnTestAPI.@logged_testset $IS_COMPOSITE verbosity begin
                Test.@test all(propertynames(learner)) do name
                    !LearnAPI.is_learner(getproperty(learner, name))
                end
            end
        end

        docstring = @doc(LearnAPI.constructor(learner)) |> string
        occursin("No documentation found", docstring) && verbosity > -1 &&
            @warn "@testapi - $_human_name "*$WARN_DOCUMENTATION

        _functions = LearnAPI.functions(learner)

        LearnTestAPI.@logged_testset $FUNCTIONS verbosity begin
            Test.@test issubset(LearnAPI.functions()[1:4], _functions)
        end

        if _is_static
            LearnTestAPI.@logged_testset $FUNCTIONS3 verbosity begin
                Test.@test :(LearnAPI.features) in _functions
            end
        end

        _tags = LearnAPI.tags(learner)
        LearnTestAPI.@logged_testset $TAGS verbosity begin
            Test.@test _tags isa Tuple
            Test.@test issubset(
                _tags,
                LearnAPI.tags(),
            )
        end

        if :(LearnAPI.predict) in _functions
            _kinds_of_proxy = LearnAPI.kinds_of_proxy(learner)
            LearnTestAPI.@logged_testset $KINDS_OF_PROXY verbosity begin
                Test.@test !isempty(_kinds_of_proxy)
                Test.@test _kinds_of_proxy isa Tuple
                Test.@test all(_kinds_of_proxy) do k
                    k isa LearnAPI.KindOfProxy
                end
            end
        end

        for (i, data) in enumerate([$(esc.(data)...)])

            verbosity > 0 && @info dataset = "@testapi - $_human_name - dataset #$i"

            if _is_static
                model =
                    LearnTestAPI.@logged_testset $FIT_IS_STATIC verbosity begin
                        LearnAPI.fit(learner; verbosity=verbosity-1)
                    end
            else
                model =
                    LearnTestAPI.@logged_testset $FIT_IS_NOT_STATIC verbosity begin
                        LearnAPI.fit(learner, data; verbosity=verbosity-1)
                    end
            end

            LearnTestAPI.@logged_testset $LEARNER verbosity begin
                Test.@test LearnAPI.learner(model) == learner
                Test.@test LearnAPI.learner(LearnAPI.strip(model)) == learner
            end

            implemented_methods = map(
                vcat(
                    InteractiveUtils.methodswith(typeof(learner), LearnAPI),
                    InteractiveUtils.methodswith(typeof(model), LearnAPI),
                )) do f
                    name = getfield(f, :name)
                    Meta.parse("LearnAPI.$name")
                end |> unique

            implemented_methods =
                intersect(implemented_methods, LearnAPI.functions())

            LearnTestAPI.@logged_testset $FUNCTIONS2 verbosity begin
                Test.@test all(f -> f in _functions, implemented_methods)
            end

            X = LearnTestAPI.@logged_testset $FEATURES verbosity begin
                LearnAPI.features(learner, data)
            end

            if !(isnothing(X))
                LearnTestAPI.@logged_testset $OBS_INVOLUTIVITY verbosity begin
                    observations = LearnAPI.obs(model, X)
                    Test.@test LearnAPI.obs(model, observations) == observations
                end
            end

            # ## TODO: check involutivity of obs(learner, _) (and is this documented?)

            model2 = LearnTestAPI.@logged_testset $SERIALIZATION verbosity begin
                small_model = LearnAPI.strip(model)
                io = IOBuffer()
                Serialization.serialize(io, small_model)
                seekstart(io)
                Serialization.deserialize(io)
            end

            if :(LearnAPI.predict) in _functions
                # get data argument for `predict`:
                if isnothing(X)
                    args = ()
                    message = $PREDICT_HAS_NO_FEATURES
                else
                    args = (X,)
                    message = $PREDICT_HAS_FEATURES
                end
                yhat = LearnTestAPI.@logged_testset message verbosity begin
                    [LearnAPI.predict(model, k, args...) for k in _kinds_of_proxy]
                    LearnAPI.predict(model, args...)
                end

                LearnTestAPI.@logged_testset $STRIP verbosity begin
                    Test.@test LearnAPI.predict(LearnAPI.strip(model), args...) == yhat
                end

                LearnTestAPI.@logged_testset $STRIP2 verbosity begin
                    Test.@test LearnAPI.predict(model2, args...) == yhat
                end

                if !isnothing(X)
                    LearnTestAPI.@logged_testset $OBS_AND_PREDICT verbosity begin
                        Test.@test all(_kinds_of_proxy) do kind
                            LearnAPI.predict(model, kind, LearnAPI.obs(model, X)) ==
                                LearnAPI.predict(model, kind, X)
                        end
                    end
                end
            end

            if :(LearnAPI.transform) in _functions
                # get data argument for `transform`:
                if isnothing(X)
                    args = ()
                    message = $TRANSFORM_HAS_NO_FEATURES
                else
                    args = (X,)
                    message = $TRANSFORM_HAS_FEATURES
                end
                W = LearnTestAPI.@logged_testset message verbosity begin
                    LearnAPI.transform(model, args...)
                end

                LearnTestAPI.@logged_testset $STRIP_TRANSFORM verbosity begin
                    Test.@test LearnAPI.transform(LearnAPI.strip(model), args...) == W
                end

                LearnTestAPI.@logged_testset $STRIP2_TRANSFORM verbosity begin
                    Test.@test LearnAPI.transform(model2, args...) == W
                end

                if !isnothing(X)
                    LearnTestAPI.@logged_testset $OBS_AND_TRANSFORM verbosity begin
                        Test.@test LearnAPI.transform(model, LearnAPI.obs(model, X)) ==
                            W
                    end
                end
            end

            if :(LearnAPI.inverse_transform) in _functions
                X2 = LearnTestAPI.@logged_testset $INVERSE_TRANSFORM verbosity begin
                    LearnAPI.inverse_transform(model, W)
                end

                LearnTestAPI.@logged_testset $STRIP_INVERSE verbosity begin
                    Test.@test LearnAPI.inverse_transform(LearnAPI.strip(model), W) ==
                        X2
                end

                LearnTestAPI.@logged_testset $STRIP2_INVERSE verbosity begin
                    Test.@test LearnAPI.inverse_transform(model2, W) == X2
                end
            end

            if !_is_static
                observations =
                    LearnTestAPI.@logged_testset $OBS_INVOLUTIVITY_FIT verbosity begin
                        observations = LearnAPI.obs(learner, data)
                        Test.@test LearnAPI.obs(learner, observations) == observations
                        observations
                    end

                LearnTestAPI.@logged_testset $OBS_AND_FIT verbosity begin
                    obsmodel = LearnAPI.fit(learner, observations; verbosity=verbosity - 1)
                    if :(LearnAPI.predict) in _functions
                        Test.@test LearnAPI.predict(model, args...) ==
                            LearnAPI.predict(obsmodel, args...)
                    end
                    if :(LearnAPI.transform) in _functions
                        Test.@test LearnAPI.transform(model, args...) ==
                            LearnAPI.transform(obsmodel, args...)
                    end
                    if :(LearnAPI.inverse_transform) in _functions
                        Test.@test LearnAPI.inverse_transform(model, W) ==
                            LearnAPI.inverse_transform(obsmodel, W)
                    end
                end

                model3 =
                    LearnTestAPI.@logged_testset $SELECTED_FOR_FIT verbosity begin
                        data3 = LearnTestAPI.learner_get(learner, data)
                        if _data_interface isa LearnAPI.RandomAccess
                            LearnAPI.fit(learner, data3; verbosity=verbosity-1)
                        else
                            nothing
                        end
                    end
            end

            if !isnothing(X)
                X3 = LearnTestAPI.@logged_testset $SELECTED verbosity begin
                    LearnTestAPI.model_get(model, X)
                end
                if _data_interface isa LearnAPI.RandomAccess
                    if :(LearnAPI.predict) in _functions
                        if _is_static
                            LearnTestAPI.@logged_testset(
                                $PREDICT_ON_SELECTIONS1,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.predict(model, X3) == yhat
                                end,
                            )
                        else
                            LearnTestAPI.@logged_testset(
                                $PREDICT_ON_SELECTIONS2,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.predict(model, X3) == yhat
                                    Test.@test @nearly LearnAPI.predict(model3, X3) == yhat
                                    Test.@test @nearly LearnAPI.predict(model3, X3) == yhat
                                end,
                            )
                        end
                    end
                    if :(LearnAPI.transform) in _functions
                        if _is_static
                            LearnTestAPI.@logged_testset(
                                $TRANSFORM_ON_SELECTIONS1,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.transform(model, X3) == W
                                end,
                            )
                        else
                            LearnTestAPI.@logged_testset(
                                $TRANSFORM_ON_SELECTIONS2,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.transform(model, X3) == W
                                    Test.@test @nearly LearnAPI.transform(model3, X3) == W
                                    Test.@test @nearly LearnAPI.transform(model3, X3) == W
                                end,
                            )
                        end
                    end
                end
            end

        end # for loop over datasets
        verbosity > 0 && @info "------ @testapi - $_human_name - tests complete ------"
    end # quote
end # macro



include("learners/static_algorithms.jl")
include("learners/regression.jl")
include("learners/ensembling.jl")
#include("learners/gradient_descent.jl")
include("learners/incremental_algorithms.jl")
include("learners/dimension_reduction.jl")

export @testapi

end # module
