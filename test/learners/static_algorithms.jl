using LearnAPI
using Tables
using Test
using LearnTestAPI
import DataFrames


# # SELECTOR

learner = LearnTestAPI.Selector(names=[:x, :w])
X = DataFrames.DataFrame(rand(3, 4), [:x, :y, :z, :w])
@testapi learner X verbosity=0

@testset "test a static transformer" begin
    model = fit(learner) # no data arguments!
    W = transform(model, X)
    @test W == DataFrames.DataFrame(Tables.matrix(X)[:,[1,4]], [:x, :w])
    @test W == transform(learner, X)
end


# # FANCY SELECTOR

learner = LearnTestAPI.FancySelector(names=[:x, :w])
X = DataFrames.DataFrame(rand(3, 4), [:x, :y, :z, :w])
@testapi learner X verbosity=0

@testset "test a variation that reports byproducts" begin
    model = fit(learner) # no data arguments!
    @test !isdefined(model, :reject)
    filtered =  DataFrames.DataFrame(Tables.matrix(X)[:,[1,4]], [:x, :w])
    @test transform(model, X) == filtered
    @test transform(learner, X) == filtered
    @test LearnTestAPI.rejected(model) == [:y, :z]
end

true
