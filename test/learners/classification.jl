using Test
using LearnTestAPI
using LearnAPI
import MLCore
using StableRNGs
import DataFrames
using Tables
import CategoricalArrays
import StatsModels: @formula
import CategoricalDistributions.pdf

# # SYNTHESIZE LOTS OF DATASETS

n = 2
rng = StableRNG(345)
# has a "hidden" level, `C`:
t = CategoricalArrays.categorical(repeat("ABA", 3n)*"CC" |> collect)[1:3n]
c, a = randn(rng, 3n), rand(rng, 3n)
y = t
Y = (; t)

# feature matrix:
x = hcat(c, a) |> permutedims

# feature tables:
X = (; c, a)
X1, X2, X3, X4, X5 = X,
Tables.rowtable(X),
Tables.dictrowtable(X),
Tables.dictcolumntable(X),
DataFrames.DataFrame(X);

# full tables:
T = (; c, t, a)
T1, T2, T3, T4, T5 = T,
    Tables.rowtable(T),
    Tables.dictrowtable(T),
    Tables.dictcolumntable(T),
    DataFrames.DataFrame(T);

# StatsModels.jl @formula:
f = @formula(t ~ c + a)


# # TESTS

learner = LearnTestAPI.ConstantClassifier()
@testapi learner (X1, y) verbosity=0
@testapi learner (X2, y) (X3, y) (X4, y) (T1, :t) (T2, :t) (T3, f) (T4, f) verbosity=0

@testset "extra tests for constant classifier" begin
    model = fit(learner, (x, y))
    @test predict(model, x) == fill('A', 3n)
    @test pdf.(predict(model, Distribution(), x), 'A') ≈ fill(2/3, 3n)
end

true
