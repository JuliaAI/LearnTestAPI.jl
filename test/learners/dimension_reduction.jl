using Test
using LearnAPI
using LearnTestAPI
using StableRNGs
using Statistics

# synthesize test data:
rng = StableRNG(123)
r = svd(rand(rng, 5, 100))
U, Vt = r.U, r.Vt
X = U*diagm([1, 2, 3, 0.01, 0.01])*Vt

learner = LearnTestAPI.TruncatedSVD(codim=2)
@testapi learner X verbosity=0

@testset "test an implementation of Truncated SVD" begin
    model = @test_logs(
        (:info, r"Singular"),
        fit(learner, X),
    )
    W = transform(model, X)
    # fit-transform in one-call:
    @test transform(learner, X; verbosity=0) == W
    outdim, indim, singular_values = LearnAPI.extras(model)
    @test singular_values[4] < 10*singular_values[1]
    X_reconstructed = inverse_transform(model, W)

    # `inverse_transform` provides approximate left-inverse to `transform`:
    @test mean(abs.(X_reconstructed - X)) < 0.001
end

true
