const ERR_BAD_LENGTH = ErrorException(
    "Actual iterator length different from `Base.length`. "
)

function get(learner_or_model, data, ::LearnAPI.RandomAccess)
    observations = LearnAPI.obs(learner_or_model, data)
    n = MLUtils.numobs(observations)
    return MLUtils.getobs(observations, 1:n)
end
function get(learner_or_model, data, ::LearnAPI.FiniteIterable)
    observations = LearnAPI.obs(learner_or_model, data)
    n = Base.length(observations)
    ret = [x for x in observations]
    length(ret) == n || throw(ERR_BAD_LENGTH)
    return ret
end
function get(learner_or_model, data, ::LearnAPI.Iterable)
    observations = LearnAPI.obs(learner_or_model, data)
    return [x for x in observations]
end

"""
    learner_get(learner, data)

*Private method.*

Extract from `LearnAPI.obs(learner, data)` all observations, using the data access API
specified by `LearnAPI.data_interface(learner)`. Used to test that the output of `data`
indeed implements the specified interface.

"""
learner_get(learner, data) =  get(learner, data, LearnAPI.data_interface(learner))

"""
    model_get(model, data)

*Private method.*

Extract from `LearnAPI.obs(model, data)` all observations, using the data access API
specified by `LearnAPI.data_interface(learner)`, where `learner =
LearnAPI.learner(model)`. Used to test that the output of `data` indeed implements the
specified interface.

"""
model_get(model, data) = get(model, data, LearnAPI.data_interface(LearnAPI.learner(model)))

macro logged_testset(message, verbosity, ex)
    quote
        $verbosity > 0 && @info $message
        try
            $(esc(ex))
        catch excptn
            if $verbosity < 1
                @info "Context of failure below: "
                @info $message
            end
            rethrow(excptn)
        end
    end
end
