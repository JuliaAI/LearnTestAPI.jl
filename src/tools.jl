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

function get(learner_or_model, data, ::LearnAPI.RandomAccess, apply)
    observations = LearnAPI.obs(learner_or_model, data) |> apply
    n = MLUtils.numobs(observations)
    return MLUtils.getobs(observations, 1:n)
end
function get(learner_or_model, data, ::LearnAPI.FiniteIterable, apply)
    observations = LearnAPI.obs(learner_or_model, data)|> apply
    n = Base.length(observations)
    ret = [x for x in observations]
    length(ret) == n || throw(ERR_BAD_LENGTH)
    return ret
end
function get(learner_or_model, data, ::LearnAPI.Iterable, apply)
    observations = LearnAPI.obs(learner_or_model, data) |> apply
    return [x for x in observations]
end

"""
    learner_get(learner, data, apply=identity)

*Private method.*

Extract from `LearnAPI.obs(learner, data)`, after applying `apply`, all observations,
using the data access API specified by `LearnAPI.data_interface(learner)`. Used to test
that the output of `data` indeed implements the specified interface.

"""
learner_get(learner, data, apply=identity) =
    get(learner, data, LearnAPI.data_interface(learner), apply)

"""
    model_get(model, data)

*Private method.*

Extract from `LearnAPI.obs(model, data)`, after applying `apply`, all observations, using
the data access API specified by `LearnAPI.data_interface(learner)`, where `learner =
LearnAPI.learner(model)`. Used to test that the output of `data` indeed implements the
specified interface.

"""
model_get(model, data, apply =identity) =
    get(model, data, LearnAPI.data_interface(LearnAPI.learner(model)), apply)


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

"""
    uniontype(x)

*Private method.*

Try to return the type of `x` "without type parameters". Otherwise, return `Union{}`.

This is a hack, but probably fine for the testing purposes of LearnTestAPI.jl.

```julia
module A
struct Foo{T}
    x::T
end
end

julia> x = A.Foo(42)
Main.A.Foo{Int}(42)

julia> uniontype(typeof(x))
Main.A.Foo
```

"""
function uniontype(T)
    modl = parentmodule(T)
    s = string(T)
    without_params = split(s, '{') |> first
    T_union = try
        ex = Meta.parse(without_params)
        modl.eval(ex)
    catch
        Union{}
    end
    T_union isa Type && return T_union
    return Union{}
end


"""
    functionswith(T)

*Private method.*

Return, as a vector of symbols, a list of LearnAPI functions which have `T` or
`LearnTestAPI.uniontype(T)` as an argument. Perhaps not a perfect catch-all; see
[`LearnTestAPI.uniontype`](@ref).

"""
function functionswith(T)
    functions = LearnAPI.eval.((LearnAPI.functions()))
    S = uniontype(T)
    implemented = vcat(
        [InteractiveUtils.methodswith(T, f) for f in functions]...,
        [InteractiveUtils.methodswith(S, f) for f in functions]...,
    ) |> unique

    map(implemented) do f
        name = getfield(f, :name)
        Meta.parse("LearnAPI.$name")
    end |> unique
end
