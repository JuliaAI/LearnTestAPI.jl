module LearnTestAPI

import LearnAPI
import Test
import Serialization
import MLUtils
import StableRNGs


# # LOGGING

ERR_UNSUPPORTED_KWARG(arg) = ArgumentError(
    "Got unsupported keyword argument `$arg`. "
)

const INFO_LOUD = "Run `LearnAPI.@testapi(learner, data..., verbosity=0)` for "*
    "silent API tests."

const INFO_QUIET = "Run `LearnAPI.@testapi(learner, data..., verbosity=1)` to debug "*
    "an API test."

const INFO_CONSTRUCTOR = """

    Testing that learner can be reconstructed from its constructors.
    [Reference](https://juliaai.github.io/LearnAPI.jl/dev/reference/#learners).

    """
const INFO_IS_COMPOSITE = """

    `LearnAPI.is_composite(learner)` returns `false`. Testing no properties of `learner`
    appear to be other
    learners. [Reference](https://juliaai.github.io/LearnAPI.jl/dev/reference/#learners).

    """
const WARN_DOCUMENTATION = """

    The learner constructor is undocumented.
    [Reference](https://juliaai.github.io/LearnAPI.jl/dev/reference/#Documentation).

    """
const INFO_FUNCTIONS = """

    Testing that `LearnAPI.functions(learner)` includes the obligatory functions.  Run
    `??LearnAPI.functions` for requirements.

    """
const INFO_TAGS = """

    Testing that `LearnAPI.tags(learner)` has correct form. Run `??LearnAPI.tags` for
    requirements.

    """
const INFO_KINDS_OF_PROXY = """

    Testing that `LearnAPI.kinds_of_proxy(learner)` is overloaded and has valid form.
    [Reference](https://juliaai.github.io/LearnAPI.jl/dev/kinds_of_target_proxy/). See also
    the `LearnAPI.predict` docstring.

    """
const INFO_FIT_IS_STATIC = """

    `LearnAPI.is_static(learner)` is `true`. Therefore attempting to call
    `fit(learner)`. Run `??LearnAPI.fit` for details.

    """
const INFO_FIT_IS_NOT_STATIC = """

    `LearnAPI.is_static(learner)` is `false`. Therefore attempting to call
    `fit(learner, data)`. Run `??LearnAPI.fit` for details.

    """
const INFO_LEARNER = """

    Attempting to call `LearnAPI.strip(model)` and to check that applying
    `LearnAPI.learner` to the result or to `model` returns `learner` in both cases. Run
    `??LearnAPI.learner` for details.

    """
const INFO_FUNCTIONS2 = """

    Check for overloaded functions missing from `LearnAPI.functions(learner)`.  Run
    `??LearnAPI.functions` for a checklist.

    """
const INFO_FEATURES = """

    Attemting to call `LearnAPI.features(learner, data)`. Run `??LearnAPI.features` for
    requirements.

    """
const INFO_PREDICT_HAS_NO_FEATURES = """

    Since `LearnAPI.features(learner, data) == nothing`, we are attempting to call
    `predict(learner, kind)` for each `kind` in `LearnAPI.kinds_of_proxy(learner)`. Run
    `??LearnAPI.predict` and `??LearnAPI.kinds_of_proxy` for context.

    """
const INFO_PREDICT_HAS_FEATURES = """

    Since `X = LearnAPI.features(learner, data)` is not `nothing`, we are attempting to
    call `predict(learner, kind, X)` for each `kind` in
    `LearnAPI.kinds_of_proxy(learner)`. Run `??LearnAPI.predict` and
    `??LearnAPI.kinds_of_proxy` for context.

    """
const INFO_STRIP = """

    Checking that `predict(model, ...)` gives the same answer if we replace `model` with
    `LearnAPI.strip(model)`.

    """
const INFO_STRIP2 = """

    Checking that `predict` or `transform` can be applied to `model` after it is stripped,
    serialized and deserialized, with no change in outcomes. Run `??LearnAPI.strip` for
    requirements.

    """



const INFO_KINDS_OF_PROXY2 = """

    `LearnAPI.predict` is apparently overloaded. Checking `LearnAPI.kinds_of_proxy
    """


# # METAPROGRAMMING HELPERS

"""
    LearaAPI.verbosity(ex)

*Private method.*

If `ex` is a specification of `verbosity`, such as `:(verbosity=1)`, then return the
specified value; otherwise, return `nothing`.

"""
verbosity(ex) = nothing
function verbosity(ex::Expr)
    if ex.head == :(=)
        ex.args[1] == :verbosity || throw(ERR_UNSUPPORTED_KWARG(ex.args[1]))
        return ex.args[2] # the actual verbosity integer
    end
    return nothing
end

"""
    LearnAPI.filter_out_verbosity(exs)

*Private method.*

Return `(filtered_exs, verbosity)` where `filtered_exs` is `exs` with any `verbosity`
specification dropped, and `verbosity` is the verbosity value (`nothing` if not
specified).

"""
function filter_out_verbosity(exs)
    verb = nothing
    exs = filter(exs) do ex
        v = verbosity(ex)
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
contracts using one or more data sets.

While the `import` commands in the example below are redundant, the packages providing the
imported modules must be available in the active package environment. In particular, if
you run `@testapi` in a package test suite, then add the modules to your package's
Project.toml file, under the `[extras]` and `[targets]` headings.

```julia
using LearnTestAPI
import LearnAPI       # these imports can be omitted
import Test
import Serialization
import MLUtils
import StableRNGs

X = (
    feautre1 = [1, 2, 3],
    feature2 = ["a", "b", "c"],
    feature3 = [10.0, 20.0, 30.0],
)

@testapi MyFeatureSelector(; features=[:feature3,]) X verbosity=1
```

# Extended help

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
    if isnothing(verbosity) || verbosity > 0
        loud = true
        @info INFO_LOUD
    else
        loud = false
        verbosity > -1 && @info INFO_QUIET
    end

    # Because they are not in the LearnAPI scope, macro hygienist will expand "Test" to
    # "LearnAPI.Test", etc. So we need a workaround:
    test = :Test
    MLUtils = :MLUtils
    Serialization = :Serialization
    t = data[1]

    quote
        import Test
        import Serialization
        import MLUtils
        import LearnAPI
        import Example

        # if $loud
        #     log(message) = @info message
        # else
        #     log(message) = nothing
        # end

        learner = $(esc(learner))
        print(learner)
        _human_name = LearnAPI.human_name(learner)
        $(esc(t)).@test true


#        Test.@testset "Implementation of LearnAPI.jl for $_human_name" begin

    #         log($(LearnAPI.INFO_CONSTRUCTOR))
    #         Test.@test LearnAPI.clone(learner) == learner

    #         if !LearnAPI.is_composite(learner)
    #             log($(LearnAPI.INFO_IS_COMPOSITE))
    #             @test all(propertynames(learner)) do name
    #                 !LearnAPI.is_learner(getproperty(learner, name))
    #             end
    #         end

    #         docstring = Base.Docs.doc(LearnAPI.constructor(learner)) |> string
    #         occursin("No documentation found", docstring) &&
    #             @warn $(LearnAPI.WARN_DOCUMENTATION)

    #         _functions = LearnAPI.functions(learner)

    #         log($(LearnAPI.INFO_FUNCTIONS))
    #         Test.@test issubset(
    #             (:(LearnAPI.fit), :(LearnAPI.learner), :(LearnAPI.strip), :(LearnAPI.obs)),
    #             _functions,
    #         )

    #         log($(LearnAPI.INFO_TAGS))
    #         _tags = LearnAPI.tags(learner)
    #         Test.@test _tags isa Tuple
    #         Test.@test issubset(
    #             _tags,
    #             LearnAPI.tags(),
    #         )
    #         Test.@test issubset(
    #             (:(LearnAPI.fit), :(LearnAPI.learner), :(LearnAPI.strip), :(LearnAPI.obs)),
    #             _functions,
    #         )

    #         if :(LearnAPI.predict) in _functions
    #             _kinds_of_proxy = LearnAPI.kinds_of_proxy(learner)
    #             log($(LearnAPI.INFO_KINDS_OF_PROXY))
    #             Test.@test !isempty(_kinds_of_proxy)
    #             Test.@test _kinds_of_proxy isa Tuple
    #             Test.@test all(_kinds_of_proxy) do k
    #                 k isa LearnAPI.KindOfProxy
    #             end
    #         end

    #         _is_static = LearnAPI.is_static(learner)

    #         for (i, data) in enumerate([$(data...)])

    #             dataset = " Dataset #$i: "
    #             if _is_static
    #                 log(dataset*$(LearnAPI.INFO_FIT_IS_STATIC))
    #                 model = LearnAPI.fit(learner)
    #             else
    #                 log(dataset*$(LearnAPI.INFO_FIT_IS_NOT_STATIC))
    #                 model = LearnAPI.fit(learner, data)
    #             end

    #             log(dataset*$(LearnAPI.INFO_LEARNER))
    #             Test.@test LearnAPI.learner(model) == learner
    #             Test.@test LearnAPI.learner(LearnAPI.strip(model)) == learner

    #             implemented_methods = map(
    #                 vcat(
    #                     methodswith(typeof(learner), LearnAPI),
    #                     methodswith(typeof(model), LearnAPI),
    #                 )) do f
    #                     name = getfield(f, :name)
    #                     Meta.parse("LearnAPI.$name")
    #                 end |> unique

    #             implemented_methods =
    #                 intersect(implemented_methods, LearnAPI.functions())

    #             log(dataset*$(LearnAPI.INFO_FUNCTIONS2))
    #             Test.@test all(f -> f in _functions, implemented_methods)

    #             log(dataset*$(LearnAPI.INFO_FEATURES))
    #             X = LearnAPI.features(learner, data)

    #             if :(LearnAPI.predict) in _functions
    #                 # get data argument for `predict`:
    #                 if isnothing(X)
    #                     args = ()
    #                     log(dataset*$(LearnAPI.INFO_PREDICT_HAS_NO_FEATURES))
    #                 else
    #                     args = (X,)
    #                     log(dataset*$(LearnAPI.INFO_PREDICT_HAS_FEATURES))
    #                 end
    #                 [LearnAPI.predict(model, k, args...) for k in _kinds_of_proxy]
    #                 yhat = LearnAPI.predict(model, args...)

    #                 log(dataset*$(LearnAPI.INFO_STRIP))
    #                 Test.@test LearnAPI.predict(LearnAPI.strip(model), args...) == yhat

    #                 log(dataset*$(LearnAPI.INFO_STRIP2))
    #                 small_model = LearnAPI.strip(model)
    #                 io = IOBuffer()
    #                 Serialization.serialize(io, small_model)
    #                 seekstart(io)
    #                 model2 = Serialization.deserialize(io)
    #                 Test.@test LearnAPI.predict(model2, args...) == yhat
    #             end
    #         end
    #    end
    end
end

end # module
