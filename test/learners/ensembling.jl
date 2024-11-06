using Test
using LearnAPI
using Random
using Statistics
import DataFrames
using StableRNGs
using Tables
using LearnTestAPI

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
@testapi learner data verbosity=0

@testset "test an implementation of bagged ensemble of ridge regressors" begin
    @test LearnAPI.clone(learner) == learner
    @test LearnAPI.target(learner, data) == y
    @test LearnAPI.features(learner, data) == X

    model = @test_logs(
        (:info, r"Trained 4 ridge"),
        fit(learner, Xtrain, y[train]; verbosity=1),
    );

    # add 3 atomic models to the ensemble:
    model = update(model, Xtrain, y[train]; verbosity=0, n=7);
    ŷ7 = predict(model, Xtest)

    # compare with cold restart:
    model_cold = fit(LearnAPI.clone(learner; n=7), Xtrain, y[train]; verbosity=0);
    @test ŷ7 ≈ predict(model_cold, Xtest)

    # test that we get a cold restart if another hyperparameter is changed:
    model2 = update(model, Xtrain, y[train]; atom=LearnTestAPI.Ridge(0.05), verbosity=0)
    learner2 = LearnTestAPI.Ensemble(LearnTestAPI.Ridge(0.05); n=7, rng)
    model_cold = fit(learner2, Xtrain, y[train]; verbosity=0)
    @test predict(model2, Xtest) ≈ predict(model_cold, Xtest)

end

true
