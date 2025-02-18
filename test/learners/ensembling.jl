using Test
using LearnAPI
using Random
using Statistics
import DataFrames
using StableRNGs
using Tables
using LearnTestAPI


# # `Ensemble`

# synthesize test data:
N = 10 # number of observations
train = 1:6
test = 7:10
a, b, c = rand(N), rand(N), rand(N)
X = (; a, b, c)
X = DataFrames.DataFrame(X)
y = 2a - b + 3c + 0.05*rand(N)
data = (X, y)
Xtrain = Tables.subset(X, train)
Xtest = Tables.subset(X, test)

rng = StableRNG(123)
atom = LearnTestAPI.Ridge()
learner = LearnTestAPI.Ensemble(atom; n=4, rng)
@testapi learner data verbosity=1

@testset "extra tests for ensemble" begin
    @test LearnAPI.clone(learner) == learner
    @test LearnAPI.target(learner, data) == y
    @test LearnAPI.features(learner, data).features == Tables.matrix(X)'

    model = @test_logs(
        (:info, r"Trained 4 ridge"),
        fit(learner, (Xtrain, y[train]); verbosity=1),
    );

    # add 3 atomic models to the ensemble:
    model = update(model, (Xtrain, y[train]), :n=>7; verbosity=0);
    ŷ7 = predict(model, Xtest)

    # compare with cold restart:
    model_cold = fit(LearnAPI.clone(learner, :n=>7), (Xtrain, y[train]); verbosity=0);
    @test ŷ7 ≈ predict(model_cold, Xtest)

    # test that we get a cold restart if another hyperparameter is changed:
    model2 = update(model, (Xtrain, y[train]), :atom=>LearnTestAPI.Ridge(0.05); verbosity=0)
    learner2 = LearnTestAPI.Ensemble(LearnTestAPI.Ridge(0.05); n=7, rng)
    model_cold = fit(learner2, (Xtrain, y[train]); verbosity=0)
    @test predict(model2, Xtest) ≈ predict(model_cold, Xtest)

end


# # `StumpRegressor`

# synthesize data:
rng = StableRNG(123)
n = 50
x = range(0, stop=2pi, length=n)[randperm(rng, n)]
y = cos.(x)

learner = LearnTestAPI.StumpRegressor(; rng=rng)
@testapi learner (x, y) verbosity=0

# Returns `true` when the `nth` last element of `losses` is better than all subsequent
# elements:
function time_to_stop(losses; n=6)
    length(losses) < n && return false
    tail = losses[end - n + 1:end]
    first(tail) == min(tail...) && return true
    false
end

@testset "extra tests for stump regressor" begin
    learner = LearnTestAPI.StumpRegressor(ntrees=1; rng=rng)
    model = fit(learner, (x, y), verbosity=0)
    # update, one tree at a time, until out-of-sample loss is not improving, in sense of
    # "number since best = 6" criterion:
    for ntrees = 2:100
        model = update(model, (x, y), :ntrees => ntrees; verbosity=0)
        time_to_stop(LearnAPI.out_of_sample_losses(model); n=6) && break
    end
    @test length(LearnAPI.trees(model)) < 100
end

true
