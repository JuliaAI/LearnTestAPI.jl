using Test
using LearnAPI
using Statistics
using StableRNGs
using LearnTestAPI
import Distributions

rng = StableRNG(123)
y = rand(rng, 50);
ynew = rand(rng, 10);

learner = LearnTestAPI.NormalEstimator()
@testapi learner y verbosity=0

@testset "NormalEstimator" begin
    model = fit(learner, y)
    d = predict(model)
    μ, σ = Distributions.params(d)
    @test μ ≈ mean(y)
    @test σ ≈ std(y)*sqrt(49/50) # `std` uses Bessel's correction

    # accessor function:
    @test LearnAPI.extras(model) == (; μ, σ)

    # one-liner:
    @test predict(learner, y) == d
    @test predict(learner, Point(), y) ≈ μ
    @test predict(learner, ConfidenceInterval(), y)[1] ≈ quantile(d, 0.025)

    # updating:
    model = update_observations(model, ynew)
    μ2, σ2 = LearnAPI.extras(model)
    μ3, σ3 = LearnAPI.extras(fit(learner, vcat(y, ynew))) # training ab initio
    @test μ2 ≈ μ3
    @test σ2 ≈ σ3
end
