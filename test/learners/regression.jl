using Test
using LearnAPI
using LinearAlgebra
using Tables
import MLUtils
import DataFrames
using LearnTestAPI

# synthesize test data:
n = 30 # number of observations
train = 1:6
test = 7:10
a, b, c = rand(n), rand(n), rand(n)
X = (; a, b, c)
X = DataFrames.DataFrame(X)
y = 2a - b + 3c + 0.05*rand(n)
data = (X, y)


# # RIDGE

learner = LearnTestAPI.Ridge(lambda=0.5)
@testapi learner data verbosity=0

@testset "extra tests for ridge regression" begin
    @test :(LearnAPI.obs) in LearnAPI.functions(learner)

    @test LearnAPI.target(learner, data) == y
    @test LearnAPI.features(learner, data) == X

    # verbose fitting:
    @test_logs(
        (:info, r"Feature"),
        fit(
            learner,
            Tables.subset(X, train),
            y[train];
            verbosity=1,
        ),
    )

    # quiet fitting:
    model = @test_logs(
        fit(
            learner,
            Tables.subset(X, train),
            y[train];
            verbosity=0,
        ),
    )

    ŷ = predict(model, Point(), Tables.subset(X, test))
    @test ŷ isa Vector{Float64}
    @test predict(model, Tables.subset(X, test)) == ŷ

    fitobs = LearnAPI.obs(learner, data)
    predictobs = LearnAPI.obs(model, X)
    model = fit(learner, MLUtils.getobs(fitobs, train); verbosity=0)
    @test LearnAPI.target(learner, fitobs) == y
    @test predict(model, Point(), MLUtils.getobs(predictobs, test)) ≈ ŷ
    @test predict(model, LearnAPI.features(learner, fitobs)) ≈ predict(model, X)

    @test LearnAPI.feature_importances(model) isa Vector{<:Pair{Symbol}}
    @test LearnAPI.feature_names(model) == [:a, :b, :c]

    filename = tempname()
    using Serialization
    small_model = LearnAPI.strip(model)
    serialize(filename, small_model)

    recovered_model = deserialize(filename)
    @test LearnAPI.learner(recovered_model) == learner
    @test predict(
        recovered_model,
        Point(),
        MLUtils.getobs(predictobs, test)
    ) ≈ ŷ

end


# # BABY RIDGE

learner = LearnTestAPI.BabyRidge(lambda=0.5)
@testapi learner data verbosity=1

@testset "extra tests for baby ridge" begin
    model = fit(learner, Tables.subset(X, train), y[train]; verbosity=0)
    ŷ = predict(model, Point(), Tables.subset(X, test))
    @test ŷ isa Vector{Float64}

    fitobs = obs(learner, data)
    predictobs = LearnAPI.obs(model, X)
    model = fit(learner, MLUtils.getobs(fitobs, train); verbosity=0)
    @test predict(model, Point(), MLUtils.getobs(predictobs, test)) == ŷ ==
        predict(model, MLUtils.getobs(predictobs, test))
    @test LearnAPI.target(learner, data) == y
    @test LearnAPI.predict(model, X) ≈
        LearnAPI.predict(model, LearnAPI.features(learner, data))
end

true
