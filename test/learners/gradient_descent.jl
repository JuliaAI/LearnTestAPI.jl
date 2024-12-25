# THIS FILE IS NOT INCLUDED BY /test/runtests.jl because of heavy dependencies.  The
# source file, "/src/learners/gradient_descent.jl" is not included in the package, but
# exits as a learner exemplar. Next line manually loads the source:
include(joinpath(@__DIR__, "..", "..", "src", "learners", "gradient_descent.jl")

using Test
using LearnAPI
using LearnTestAPI
using Random
using Statistics
using StableRNGs
import Optimisers
import Zygote
import NNlib
import CategoricalDistributions
import CategoricalDistributions: pdf, mode
import ComponentArrays

# synthetic test data:
N = 10
n = 10N # number of observations
p = 2   # number of features
train = 1:6N
test = (6N+1:10N)
rng = StableRNG(123)
X = randn(rng, Float32, p, n);
coefficients = rand(rng, Float32, p)'
y_continuous = coefficients*X |> vec
η1 = quantile(y_continuous, 1/3)
η2 = quantile(y_continuous, 2/3)
y = map(y_continuous) do η
    η < η1 && return "A"
    η < η2 && return "B"
    "C"
end |> CategoricalDistributions.categorical;
Xtrain = X[:, train];
Xtest = X[:, test];
ytrain = y[train];
ytest = y[test];

rng = StableRNG(123)
learner =
    PerceptronClassifier(; optimiser=Optimisers.Adam(0.01), epochs=40, rng)

@testapi learner (X, y) verbosity=1

@testset "PerceptronClassfier" begin
    @test LearnAPI.clone(learner) == learner
    @test :(LearnAPI.update) in LearnAPI.functions(learner)
    @test LearnAPI.target(learner, (X, y)) == y
    @test LearnAPI.features(learner, (X, y)) == X

    model40 = fit(learner, Xtrain, ytrain; verbosity=0)

    # 40 epochs is sufficient for 90% accuracy in this case:
    @test sum(predict(model40, Point(), Xtest) .== ytest)/length(ytest) > 0.9

    # get probabilistic predictions:
    ŷ40 = predict(model40, Distribution(), Xtest);
    @test predict(model40, Xtest) ≈ ŷ40

    # add 30 epochs in an `update`:
    model70 = update(model40, Xtrain, y[train]; verbosity=0, epochs=70)
    ŷ70 = predict(model70, Xtest);
    @test !(ŷ70 ≈ ŷ40)

    # compare with cold restart:
    model = fit(LearnAPI.clone(learner; epochs=70), Xtrain, y[train]; verbosity=0);
    @test ŷ70 ≈ predict(model, Xtest)

    # instead add 30 epochs using `update_observations` instead:
    model70b = update_observations(model40, Xtrain, y[train]; verbosity=0, epochs=30)
    @test ŷ70 ≈ predict(model70b, Xtest) ≈ predict(model, Xtest)
end

true
