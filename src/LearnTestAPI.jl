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
    join(map(ex->"`:$ex`", LearnAPI.functions()[1:5]), ", ", " and ")
const FUNCTIONS = """

    Testing that `LearnAPI.functions(learner)` includes $OBLIGATORY_FUNCTIONS.

  """
const ERR_MISSINNG_OBLIGATORIES =
    "These obligatory functions are missing from the return value of "*
    "`LearnAPI.functions(learner)`; "

const FUNCTIONS3 = """

    Testing that `LearnAPI.functions(learner)` includes `:(LearnAPI.features).`

  """
const FUNCTIONS4 = """

    Testing that `LearnAPI.functions(learner)` exludes `:(LearnAPI.features)`, as
    `LearnAPI.is_static(learner)` is `true`.

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

    Checking for overloaded LearnAPI.jl functions missing from the return value of
    `LearnAPI.functions(learner)`. Run `??LearnAPI.functions` for a
    checklist.

  """
const ERR_MISSING_FUNCTIONS =
    "The following overloaded functions are missing from the return value of"*
    "`LearnAPI.functions(learner)`: "

const OBS = """

    Attempting to call `observations = obs(learner, data)`.

  """
const FEATURES = """

    Attempting to call `LearnAPI.features(learner, observations)`.

  """
const OBS_INVOLUTIVITY = """

    Testing that `obs(model, obs(model, data)) == obs(model, data)` (involutivity).

  """
const SERIALIZATION = """

    Checking that `LearnAPI.strip(model)` can be Julia-serialized/deserialized.

  """
const PREDICT_HAS_NO_FEATURES = """

    Attempting to call `predict(model, kind)` for each `kind` in
    `LearnAPI.kinds_of_proxy(learner)`. (We are not providing `predict` with a data
    argument because `features(obs(learner, data)) == nothing`).


  """
const PREDICT_HAS_FEATURES = """

    Attempting to call `predict(model, kind, X)` for each `kind` in
    `LearnAPI.kinds_of_proxy(learner)`. Here `X = LearnAPI.features(obs(learner, data))`.

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

    Attempting to call `transform(model)` (and not `transform(model, X)`, because
    `features(obs(learner, data)) == nothing`).

  """
const TRANSFORM_HAS_FEATURES = """

    Attempting to call `transform(model, X)`.

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
    `transform(model, X)` and `X = LearnAPI.features(learner, observations)`.

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
    testing that we can call `fit` on the selection, obtaining a new `fit` result
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
const TARGET = """

    Attempting to call `LearnAPI.target(learner, observations)` (fallback returns
    `nothing`).

  """
const TARGET_IN_FUNCTIONS = """

    Checking that `:(LearnAPI.target)` is included in `LearnAPI.functions(learner)`.

  """
const TARGET_NOT_IN_FUNCTIONS = """

    Checking that `:(LearnAPI.target)` is excluded from `LearnAPI.functions(learner)`, as
    `LearnAPI.target` has not been overloaded.

  """
const TARGET_SELECTIONS = """

    Checking that all observations can be extracted from `LearnAPI.target(learner,
    observations)` using the data interface declared by
    `LearnAPI.data_interface(learner)`.

  """
const WEIGHTS = """

    Attempting to call `LearnAPI.weights(learner, observations)` (fallback returns
    `nothing`).

  """
const WEIGHTS_IN_FUNCTIONS = """

    Checking that `:(LearnAPI.weights)` is included in `LearnAPI.functions(learner)`.

  """
const WEIGHTS_SELECTIONS = """

    Checking that all observations can be extracted from `LearnAPI.weights(learner,
    observations)` using the data interface declared by
    `LearnAPI.data_interface(learner)`.

  """
const WEIGHTS_NOT_IN_FUNCTIONS = """

    Checking that `:(LearnAPI.weights)` is excluded from `LearnAPI.functions(learner)`, as
    `LearnAPI.weights` has not been overloaded.

  """
const COEFFICIENTS = """

    Checking `LearnAPI.coefficients(model)` has the correct form, and that the feature
    names match those returned by `LearnAPI.feature_names(model)`, if the latter is
    implemented.

  """
const UPDATE = """

    Calling `LearnAPI.update(model, data; verbosity=0)`.

  """
const ERR_STATIC_UPDATE = ErrorException(
    "`(LearnAPI.update)` is in `LearnAPI.functions(learner)` but "*
        "`LearnAPI.is_static(learner)` is `true`. You cannot implement `update` "*
        "for static learners. "
)
const UPDATE_ITERATIONS = """

    Attempting to increase number of iterations by one with an `update` call, giving a new
    model `newmodel`. Checking that `LearnAPI.learner(newmodel)` reflects the iteration
    update. Checking that retraining from scratch with the extra iteration gives,
    approximately, the same predictions/transformations.

  """
const ERR_BAD_UPDATE = ErrorException(
"Updating iterations by one and instead retraining from scratch are giving "*
    "different outcomes. "
)
const INTERCEPT = """

    Checking `LearnAPI.intercept(model)` has the correct form.

  """
const FEATURE_IMPORTANCES = """

    Checking `LearnAPI.feature_importances(model)` has the correct form, and that the
    feature names match those returned by `LearnAPI.feature_names(model)`, if the
    latter is implemented.

  """
const TRAINING_LOSSES = """

    Checking that `LearnAPI.training_losses(model)` is a vector.

  """
const OUT_OF_SAMPLE_LOSSES = """

    Checking that `LearnAPI.out_of_sample_losses(model)` is a vector.

  """
const TRAINING_SCORES = """

    Checking that `LearnAPI.training_scores(model)` is a vector.

  """
const PREDICTIONS = """

    Checking that `LearnAPI.predictions(model)` and `predict(model, X)` are approximately
    equal.

  """
const ERR_STATIC_PREDICTIONS = ErrorException(
    "`LearnAPI.functions(learner)` includes `:(LearnAPI.predictions)` but you may "*
    "not implement `LearnAPI.predictions` for static learners. "
)

const OUT_OF_SAMPLE_INDICES = """

    Checking that `LearnAPI.out_of_sample_indices(model)` is a vector of integers.

  """
const ERR_OOS_INDICES = ErrorException(
    "`:(LearnAPI.out_of_sample_indices)` is in the return value of "*
        "`LearnAPI.functions(learner)` but `:(LearnAPI.predictions)` is not."
)


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
        import LearnTestAPI.Test
        import LearnTestAPI.Serialization
        import LearnTestAPI.MLUtils
        import LearnTestAPI.LearnAPI
        import LearnTestAPI.InteractiveUtils
        import LearnTestAPI: @logged_testset, @nearly

        learner = $(esc(learner))
        verbosity=$verbosity
        _human_name = LearnAPI.human_name(learner)
        _data_interface = LearnAPI.data_interface(learner)
        _is_static = LearnAPI.is_static(learner)

        if isnothing(verbosity) || verbosity > 0
            @info "------ running @testapi - $_human_name "*$LOUD
        else
            verbosity > -1 && @info "running @testapi - $_human_name "*$QUIET
        end

        @logged_testset $CONSTRUCTOR verbosity begin
            Test.@test LearnAPI.clone(learner) == learner
        end

        if !LearnAPI.is_composite(learner)
            @logged_testset $IS_COMPOSITE verbosity begin
                Test.@test all(propertynames(learner)) do name
                    !LearnAPI.is_learner(getproperty(learner, name))
                end
            end
        end

        docstring = @doc(LearnAPI.constructor(learner)) |> string
        occursin("No documentation found", docstring) && verbosity > -1 &&
            @warn "@testapi - $_human_name "*$WARN_DOCUMENTATION

        _functions = LearnAPI.functions(learner)

        @logged_testset $FUNCTIONS verbosity begin
            awol = setdiff(LearnAPI.functions()[1:5], _functions)
            if isempty(awol)
                Test.@test true
            else
                @error ERR_MISSINNG_OBLIGATORIES*string(awol)
                Test.@test false
            end
        end

        if !_is_static
            @logged_testset $FUNCTIONS3 verbosity begin
                Test.@test :(LearnAPI.features) in _functions
            end
        else
            @logged_testset $FUNCTIONS4 verbosity begin
                Test.@test !(:(LearnAPI.features) in _functions)
            end
        end

        _tags = LearnAPI.tags(learner)
        @logged_testset $TAGS verbosity begin
            Test.@test _tags isa Tuple
            Test.@test issubset(
                _tags,
                LearnAPI.tags(),
            )
        end

        if :(LearnAPI.predict) in _functions
            _kinds_of_proxy = LearnAPI.kinds_of_proxy(learner)
            @logged_testset $KINDS_OF_PROXY verbosity begin
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
                    @logged_testset $FIT_IS_STATIC verbosity begin
                        LearnAPI.fit(learner; verbosity=verbosity-1)
                    end
            else
                model =
                    @logged_testset $FIT_IS_NOT_STATIC verbosity begin
                        LearnAPI.fit(learner, data; verbosity=verbosity-1)
                    end
            end

            @logged_testset $LEARNER verbosity begin
                Test.@test LearnAPI.learner(model) == learner
                Test.@test @nearly LearnAPI.learner(LearnAPI.strip(model)) == learner
            end

            # try to catch as many implemented methods as possible:
            implemented = vcat(
                LearnTestAPI.functionswith(typeof(learner)),
                LearnTestAPI.functionswith(typeof(model)),
            ) |> unique

            @logged_testset $FUNCTIONS2 verbosity begin
                awol = setdiff(implemented, _functions)
                if isempty(awol)
                    Test.@test true
                else
                    @error ERR_MISSING_FUNCTIONS*string(awol)
                    Test.@test false
                end
            end

            observations = @logged_testset $OBS verbosity begin
                obs(learner, data)
            end

            X = @logged_testset $FEATURES verbosity begin
                LearnAPI.features(learner, observations)
            end

            if !(isnothing(X))
                @logged_testset $OBS_INVOLUTIVITY verbosity begin
                    _observations = LearnAPI.obs(model, X)
                    Test.@test LearnAPI.obs(model, _observations) == _observations
                end
            end

            model2 = @logged_testset $SERIALIZATION verbosity begin
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
                yhat = @logged_testset message verbosity begin
                    [LearnAPI.predict(model, k, args...) for k in _kinds_of_proxy]
                    LearnAPI.predict(model, args...)
                end

                @logged_testset $STRIP verbosity begin
                    Test.@test @nearly(
                    LearnAPI.predict(LearnAPI.strip(model), args...) == yhat,
                    )
                end

                @logged_testset $STRIP2 verbosity begin
                    Test.@test @nearly LearnAPI.predict(model2, args...) == yhat
                end

                if !isnothing(X)
                    @logged_testset $OBS_AND_PREDICT verbosity begin
                        Test.@test all(_kinds_of_proxy) do kind
                            @nearly(
                                LearnAPI.predict(model, kind, LearnAPI.obs(model, X)) ==
                                    LearnAPI.predict(model, kind, X),
                            )
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
                W = @logged_testset message verbosity begin
                    LearnAPI.transform(model, args...)
                end

                @logged_testset $STRIP_TRANSFORM verbosity begin
                    Test.@test @nearly(
                    LearnAPI.transform(LearnAPI.strip(model), args...) == W,
                    )
                end

                @logged_testset $STRIP2_TRANSFORM verbosity begin
                    Test.@test @nearly LearnAPI.transform(model2, args...) == W
                end

                if !isnothing(X)
                    @logged_testset $OBS_AND_TRANSFORM verbosity begin
                        Test.@test @nearly(
                            LearnAPI.transform(model, LearnAPI.obs(model, X)) == W,
                        )
                    end
                end
            end

            if :(LearnAPI.inverse_transform) in _functions && !isnothing(X)
                X2 = @logged_testset $INVERSE_TRANSFORM verbosity begin
                    LearnAPI.inverse_transform(model, W)
                end

                @logged_testset $STRIP_INVERSE verbosity begin
                    Test.@test @nearly(
                        LearnAPI.inverse_transform(LearnAPI.strip(model), W) == X2,
                    )
                end

                @logged_testset $STRIP2_INVERSE verbosity begin
                    Test.@test @nearly LearnAPI.inverse_transform(model2, W) == X2
                end
            end

            if !_is_static
                _observations =
                    @logged_testset $OBS_INVOLUTIVITY_FIT verbosity begin
                        _observations = LearnAPI.obs(learner, data)
                        Test.@test LearnAPI.obs(learner, _observations) == _observations
                        _observations
                    end

                @logged_testset $OBS_AND_FIT verbosity begin
                    obsmodel =
                        LearnAPI.fit(learner, _observations; verbosity=verbosity - 1)
                    if :(LearnAPI.predict) in _functions
                        Test.@test @nearly(
                            LearnAPI.predict(model, args...) ==
                                LearnAPI.predict(obsmodel, args...),
                        )
                    end
                    if :(LearnAPI.transform) in _functions
                        Test.@test @nearly(
                            LearnAPI.transform(model, args...) ==
                                LearnAPI.transform(obsmodel, args...),
                        )
                    end
                    if :(LearnAPI.inverse_transform) in _functions
                        Test.@test @nearly(
                            LearnAPI.inverse_transform(model, W) ==
                                LearnAPI.inverse_transform(obsmodel, W),
                        )
                    end
                end

                model3 =
                    @logged_testset $SELECTED_FOR_FIT verbosity begin
                        data3 = LearnTestAPI.learner_get(learner, data)
                        if _data_interface isa LearnAPI.RandomAccess
                            LearnAPI.fit(learner, data3; verbosity=verbosity-1)
                        else
                            nothing
                        end
                    end
            end

            if !isnothing(X)
                X3 = @logged_testset $SELECTED verbosity begin
                    LearnTestAPI.model_get(model, X)
                end
                if _data_interface isa LearnAPI.RandomAccess
                    if :(LearnAPI.predict) in _functions
                        if _is_static
                            @logged_testset(
                                $PREDICT_ON_SELECTIONS1,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.predict(model, X3) == yhat
                                end,
                            )
                        else
                            @logged_testset(
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
                            @logged_testset(
                                $TRANSFORM_ON_SELECTIONS1,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.transform(model, X3) == W
                                end,
                            )
                        else
                            @logged_testset(
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
                elseif _data_interface isa LearnAPI.FiniteIterable
                    if :(LearnAPI.predict) in _functions
                        if _is_static
                            @logged_testset(
                                $PREDICT_ON_SELECTIONS1,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.predict(model, X3) == yhat
                                end,
                            )
                        else
                            @logged_testset(
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
                            @logged_testset(
                                $TRANSFORM_ON_SELECTIONS1,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.transform(model, X3) == W
                                end,
                            )
                        else
                            @logged_testset(
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

            # target

            _y = @logged_testset $TARGET verbosity begin
                LearnAPI.target(learner, observations)
            end

            if !(isnothing(_y))
                @logged_testset $TARGET_IN_FUNCTIONS verbosity begin
                    Test.@test :(LearnAPI.target) in _functions
                end
                y = @logged_testset $TARGET_SELECTIONS verbosity begin
                    LearnTestAPI.learner_get(
                        learner,
                        data,
                        data->LearnAPI.target(learner, data),
                    )
                end
            else
                @logged_testset $TARGET_NOT_IN_FUNCTIONS verbosity begin
                    Test.@test !(:(LearnAPI.target) in _functions)
                end
            end

            # weights

            _w = @logged_testset $WEIGHTS verbosity begin
                LearnAPI.weights(learner, observations)
            end

            if !(isnothing(_w))
                @logged_testset $WEIGHTS_IN_FUNCTIONS verbosity begin
                    Test.@test :(LearnAPI.weights) in _functions
                end
                w = @logged_testset $WEIGHTS_SELECTIONS verbosity begin
                    LearnTestAPI.learner_get(
                        learner,
                        data,
                        data->LearnAPI.weights(learner, data),
                    )
                end
            else
                @logged_testset $WEIGHTS_NOT_IN_FUNCTIONS verbosity begin
                    Test.@test !(:(LearnAPI.weights) in _functions)
                end
            end

            # update

            if :(LearnAPI.update) in _functions
                _is_static && throw($ERR_STATIC_UPDATE)
                @logged_testset $UPDATE verbosity begin
                    LearnAPI.update(model, data; verbosity=0)
                end
                # only test hyperparameter replacement in case of iteration parameter:
                iter = LearnAPI.iteration_parameter(learner)
                if !isnothing(iter)
                    @logged_testset $UPDATE_ITERATIONS verbosity begin
                        n = getproperty(learner, iter)
                        newmodel = LearnAPI.update(model, data, iter=>n+1; verbosity=0)
                        newlearner = LearnAPI.clone(learner, iter=>n+1)
                        Test.@test LearnAPI.learner(newmodel) == newlearner
                        abinitiomodel = LearnAPI.fit(newlearner, data; verbosity=0)
                        # prepare to test `predict` and `transform` are the same for ab
                        # initio model and for updated model:
                        if isnothing(X)
                            args = ()
                        else
                            args = (X,)
                        end
                        operations = intersect(
                            _functions,
                            [:(LearnAPI.predict), :(LearnAPI.transform)],
                        )
                        for ex in operations
                            op = eval(:($ex))
                            left = op(newmodel, args...)
                            right = op(newmodel, args...)
                            agree = @nearly left == right
                            if agree
                                Test.@test true
                            else
                                @error $ERR_BAD_UPDATE
                            end
                        end
                    end
                end
            end

            # accessor functions
            # `learner`, `extras`, `strip`, `already` tested above

            if :(LearnAPI.coefficients) in _functions
                @logged_testset $COEFFICIENTS verbosity begin
                    coefs = LearnAPI.coefficients(model)
                    Test.@test coeffs isa AbstractVector{
                        <:Pair{<:Any,<:Union{Real,AbstractVector{<:Real}}}
                    }
                    if :(LearnAPI.feature_names) in _functions
                        Test.@test Set(LearnAPI.feature_names(model)) ==
                            Set(first.(coeffs))
                    end
                end
            end

            if :(LearnAPI.intercept) in _functions
                @logged_testset $INTERCEPT verbosity begin
                    c = LearnAPI.intercept(model)
                    Test.@test c isa Union{Real,AbstractVector{<:Real}}
                end
            end

            if :(LearnAPI.feature_importances) in _functions
                @logged_testset $FEATURE_IMPORTANCES verbosity begin
                    fis = LearnAPI.feature_importances(model)
                    Test.@test fis isa AbstractVector{
                        <:Pair{<:Any,<:Real}
                    }
                    if :(LearnAPI.feature_names) in _functions
                        Test.@test Set(LearnAPI.feature_names(model)) ==
                            Set(first.(fis))
                    end
                end
            end

            if :(LearnAPI.training_losses) in _functions
                @logged_testset $TRAINING_LOSSES verbosity begin
                    losses = LearnAPI.training_losses(model)
                    Test.@test losses isa AbstractVector
                end
            end

            if :(LearnAPI.out_of_sample_losses) in _functions
                @logged_testset $OUT_OF_SAMPLE_LOSSES verbosity begin
                    losses = LearnAPI.out_of_sample_losses(model)
                    Test.@test losses isa AbstractVector
                end
            end

            if :(LearnAPI.training_scores) in _functions
                @logged_testset $TRAINING_SCORES verbosity begin
                    losses = LearnAPI.training_scores(model)
                    Test.@test losses isa AbstractVector
                end
            end

            if :(LearnAPI.predictions) in _functions
                _is_static && throw(ERR_STATIC_PREDICTIONS)
                @logged_testset $PREDICTIONS verbosity begin
                    Test.@test @nearly(
                        LearnAPI.predictions(model) == LearnAPI.predict(model, X),
                    )
                end
            end

            if :(LearnAPI.out_of_sample_indices) in _functions
                :(LearnAPI.predictions) in _functions || throw(ERR_OOS_INDICES)
                @logged_testset $OUT_OF_SAMPLE_INDICES verbosity begin
                    Test.@test(
                        LearnAPI.out_of_sample_indices(model) isa AbstractVector{<:Integer}
                    )
                end
            end




        end # for loop over datasets
        verbosity > 0 && @info "------ @testapi - $_human_name - tests complete ------"
        nothing
    end # quote
end # macro

include("learners/static_algorithms.jl")
include("learners/regression.jl")
include("learners/ensembling.jl")
# next learner excluded because of heavy dependencies:
# include("learners/gradient_descent.jl")
include("learners/incremental_algorithms.jl")
include("learners/dimension_reduction.jl")

export @testapi

end # module
