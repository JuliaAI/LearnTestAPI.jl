"""
    LearnTestAPI.@logged_testset message verbosity ex

*Private method.*

Similar to `Test.@testset` with the exception that you can make `message` longer without
loss of clarity, and this message is reported according to the extra parameter `verbosity`
provided:

- If `verbosity > 0` then `message` is always logged to `Logging.Info`

- If `verbosity ≤ 0` then `message` is only logged if a `@test` call in `ex` fails on
  evaluation.

Note that variables defined within the program `ex` are local, and so are not available in
subsequent `@logged_testset` calls. However, the last evaluated expression in `ex` is the
return value.

```julia-repl
julia> f(x) = x^2
julia> LearnTestAPI.@logged_testset "Testing `f`" 1 begin
           y = f(2)
           @test y > 0
           y
       end
[ Info: Testing `f`
4

julia> LearnTestAPI.@logged_testset "Testing `f`" 0 begin
           y = f(2)
           @test y < 0
           y
       end
Test Failed at REPL[41]:3
  Expression: y < 0
   Evaluated: 4 < 0

[ Info: Context of failure:
[ Info: Testing `f`
ERROR: There was an error during testing
```

"""
macro logged_testset(message, verbosity, ex)
    quote
        $verbosity > 0 && @info $message
        try
            $ex
        catch excptn
            if $verbosity < 1
                @info "Context of failure: "
                @info $message
            end
            rethrow(excptn)
        end
    end |> esc
end

const ERR_BAD_LENGTH = ErrorException(
    "Actual iterator length different from `Base.length`. "
)

function get(learner_or_model, data, ::LearnAPI.RandomAccess)
    observations = LearnAPI.obs(learner_or_model, data)
    n = MLUtils.numobs(observations)
    return MLUtils.getobs(observations, 1:n)
end
function get(learner_or_model, data, ::LearnAPI.FiniteIterable)
    observations = LearnAPI.obs(learner_or_model, data)
    n = Base.length(observations)
    ret = [x for x in observations]
    length(ret) == n || throw(ERR_BAD_LENGTH)
    return ret
end
function get(learner_or_model, data, ::LearnAPI.Iterable)
    observations = LearnAPI.obs(learner_or_model, data)
    return [x for x in observations]
end

"""
    learner_get(learner, data)

*Private method.*

Extract from `LearnAPI.obs(learner, data)` all observations, using the data access API
specified by `LearnAPI.data_interface(learner)`. Used to test that the output of `data`
indeed implements the specified interface.

"""
learner_get(learner, data) =  get(learner, data, LearnAPI.data_interface(learner))

"""
    model_get(model, data)

*Private method.*

Extract from `LearnAPI.obs(model, data)` all observations, using the data access API
specified by `LearnAPI.data_interface(learner)`, where `learner =
LearnAPI.learner(model)`. Used to test that the output of `data` indeed implements the
specified interface.

"""
model_get(model, data) = get(model, data, LearnAPI.data_interface(LearnAPI.learner(model)))


const INFO_NEAR = "Tried testing for `≈` because `==` failed. "

"""
    isnear(x, y; kwargs...)

*Private method.*

Returns `true` if `x == y`. Otherwise try to return `isapprox(x, y; kwargs...)`.

```julia
julia> near('a', 'a')
true

julia> near('a', 'b')
[ Info: Tried testing for `≈` because `==` failed.
ERROR: MethodError: no method matching isapprox(::Char, ::Char)
```

"""
function isnear(x, y; kwargs...)
    x == y && return true
    try
        isapprox(x, y; kwargs...)
    catch excptn
        excptn isa MethodError && @info INFO_NEAR
        rethrow(excptn)
    end
end

"""
    @nearly lhs == rhs kwargs...

*Private method.*

Replaces the expresion `lhs == rhs` with `isnear(lhs, rhs; kwargs...)` for testing a weaker
form of equality. Here `kwargs...` are keyword arguments accepted in `isapprox(lhs, rhs;
kwargs...)`, which is called if `lhs == rhs` fails.

See also [`LearnTestAPI.isnear`](@ref).

"""
macro nearly(ex, kwarg_exs...)
    MacroTools.@capture(ex, lhs_ == rhs_) ||
        error("Usage: @nearly lhs == rhs isapprox_keyword_assignments" )
    quote
        LearnTestAPI.isnear($lhs, $rhs; $(kwarg_exs...))
    end |> esc
end
