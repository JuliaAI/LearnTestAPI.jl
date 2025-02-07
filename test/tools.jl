using Test
using LearnTestAPI
using LearnAPI

struct RandomHorse end
struct FiniteHorse end
struct IterableHorse end

const Horse = Union{RandomHorse,FiniteHorse,IterableHorse}
LearnAPI.fit(learner::Horse, data; verbosity=1) = Ref(learner)
LearnAPI.learner(model::Base.RefValue{<:Horse}) = model[]

struct Data
    v::Vector{Float64}
end

LearnAPI.obs(::Union{Horse,Base.RefValue{<:Horse}},  data::Data) = data.v
LearnAPI.obs(::Union{Horse,Base.RefValue{<:Horse}}, v::AbstractVector) = v

@trait FiniteHorse data_interface=LearnAPI.FiniteIterable()
@trait IterableHorse data_interface=LearnAPI.Iterable()

v = [1, 42, 12]
data = Data(v)
@test all([RandomHorse(), FiniteHorse(), IterableHorse()]) do learner
    model = fit(learner, data)
    LearnTestAPI.learner_get(learner, data) == v &&
        LearnTestAPI.model_get(model, data) == v
end

@test_logs(
    (:info, LearnTestAPI.INFO_NEAR),
    (@test_throws MethodError LearnTestAPI.isnear('a', 'b')),
)

true
