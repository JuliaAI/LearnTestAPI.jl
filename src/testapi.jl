"""
    @testapi learner dataset1 dataset2 ... verbosity=1

Test that `learner` correctly implements the LearnAPI.jl interface, by checking
contracts against one or more data sets.

```julia
using LearnTestAPI

X = (
    feature1 = [1, 2, 3],
    feature2 = ["a", "b", "c"],
    feature3 = [10.0, 20.0, 30.0],
)

@testapi MyFeatureSelector(; features=[:feature3,]) X verbosity=1
```

# Extended help

# Assumptions

In some tests strict `==` is enforced on the output of `predict` or `transform`, unless
`isapprox` is also implemented. If `predict` outputs categorical vectors, for example,
then requiring `==` in a test is appropriate. On the other hand, if `predict` outputs some
abstract vector of eltype `Float32`, it will be necessary that `isapprox` is implemented
for that vector type, because the strict test `==` is likely to fail. These comments apply
to more complicated objects, such as probability distributions or sampleable objects: If
`==` is likely to fail in "benign" cases, be sure `isapprox` is implemented. See
[`LearnTestAPI.isnear`](@ref) for the exact test applied.

# What is not tested?

When `verbosity=1` (the default) the test log describes all contracts tested.

The following are *not* tested:

- That the output of `LearnAPI.target(learner, data)` is indeed a target, in the sense
  that it can be paired, in some way, with the output of `predict`. Such a test would be
  to suitably pair the output with a predicted proxy for the target, using, for example, a
  proper scoring rule, in the case of probabilistic predictions.

- That `inverse_transform` is an approximate left or right inverse to `transform`

- That the one-line convenience methods, `transform(learner, ...)` or `predict(learner,
  ...)`, where implemented, have the same effect as the two-line calls they combine.

- The veracity of `LearnAPI.is_pure_julia(learner)`.

- The second of the two contracts appearing in the
  [`LearnAPI.target_observation_scitype`](@ref) docstring. The first contract is only
  tested if `LearnAPI.data_interface(learner)` is `LearnAPI.RandomAccess()` or
  `LearnAPI.FiniteIterable()`.

Whenever the internal `learner` algorithm involves case distinctions around data or
hyperparameters, it is recommended that multiple datasets, and learners with a variety of
hyperparameter settings, are explicitly tested.

# Role of datasets in tests

Each `dataset` is used as follows.

If `LearnAPI.is_static(learner) == false`, then:

- `dataset` is passed to `fit` and, if necessary, its `update` cousins

- If `X = LearnAPI.features(learner, dataset) == nothing`, then `predict` and/or
  `transform` are called with no data. Otherwise, they are called with `X`.

If instead `LearnAPI.is_static(learner) == true`, then `fit` and its cousins are called
without any data, and `dataset` is passed directly to `predict` and/or `transform`.

"""
macro testapi(learner, data...)

    data, verbosity = filter_out_verbosity(data)

    quote
        import LearnTestAPI.Test
        import LearnTestAPI.Serialization
        import LearnTestAPI.MLCore
        import LearnTestAPI.LearnAPI
        import LearnTestAPI.InteractiveUtils
        import LearnTestAPI: @logged_testset, @nearly

        learner = $(esc(learner))
        verbosity=$verbosity
        _human_name = LearnAPI.human_name(learner)
        _data_interface = LearnAPI.data_interface(learner)
        _is_static = LearnAPI.is_static(learner)

        if isnothing(verbosity) || verbosity > 0
            @info "------ running @testapi - $_human_name "*$LOUD
        else
            verbosity > -1 && @info "running @testapi - $_human_name "*$QUIET
        end

        @logged_testset $CONSTRUCTOR verbosity begin
            Test.@test LearnAPI.clone(learner) == learner
        end

        @logged_testset $NONLEARNERS verbosity begin
            Test.@test all(LearnAPI.nonlearners(learner)) do name
                !LearnAPI.is_learner(getproperty(learner, name))
            end
            Test.@test all(LearnAPI.learners(learner)) do name
                LearnAPI.is_learner(getproperty(learner, name))
            end
        end

        docstring = @doc(LearnAPI.constructor(learner)) |> string
        occursin("No documentation found", docstring) && verbosity > -1 &&
            @warn "@testapi - $_human_name "*$WARN_DOCUMENTATION

        _functions = LearnAPI.functions(learner)

        @logged_testset $FUNCTIONS verbosity begin
            awol = setdiff(LearnAPI.functions()[1:5], _functions)
            if isempty(awol)
                Test.@test true
            else
                @error ERR_MISSINNG_OBLIGATORIES*string(awol)
                Test.@test false
            end
        end

        if !_is_static
            @logged_testset $FUNCTIONS3 verbosity begin
                Test.@test :(LearnAPI.features) in _functions
            end
        else
            @logged_testset $FUNCTIONS4 verbosity begin
                Test.@test !(:(LearnAPI.features) in _functions)
            end
        end

        _tags = LearnAPI.tags(learner)
        @logged_testset $TAGS verbosity begin
            Test.@test _tags isa Tuple
            Test.@test issubset(
                _tags,
                LearnAPI.tags(),
            )
        end

        if :(LearnAPI.predict) in _functions
            _kinds_of_proxy = LearnAPI.kinds_of_proxy(learner)
            @logged_testset $KINDS_OF_PROXY verbosity begin
                Test.@test !isempty(_kinds_of_proxy)
                Test.@test _kinds_of_proxy isa Tuple
                Test.@test all(_kinds_of_proxy) do k
                    k isa LearnAPI.KindOfProxy
                end
            end
        end

        missing_traits = Set{Expr}()

        for (i, data) in enumerate([$(esc.(data)...)])

            verbosity > 0 && @info dataset = "@testapi - $_human_name - dataset #$i"

            if _is_static
                model =
                    @logged_testset $FIT_IS_STATIC verbosity begin
                        LearnAPI.fit(learner; verbosity=verbosity-1)
                    end
            else
                model =
                    @logged_testset $FIT_IS_NOT_STATIC verbosity begin
                        LearnAPI.fit(learner, data; verbosity=verbosity-1)
                    end
            end

            @logged_testset $LEARNER verbosity begin
                Test.@test LearnAPI.learner(model) == learner
                Test.@test @nearly LearnAPI.learner(LearnAPI.strip(model)) == learner
            end

            # try to catch as many implemented methods as possible:
            implemented = vcat(
                LearnTestAPI.functionswith(typeof(learner)),
                LearnTestAPI.functionswith(typeof(model)),
            ) |> unique

            @logged_testset $FUNCTIONS2 verbosity begin
                awol = setdiff(implemented, _functions)
                if isempty(awol)
                    Test.@test true
                else
                    @error ERR_MISSING_FUNCTIONS*string(awol)
                    Test.@test false
                end
            end

            observations = @logged_testset $OBS verbosity begin
                obs(learner, data)
            end

            X = if _is_static
                data
            else
                @logged_testset $FEATURES0 verbosity begin
                    LearnAPI.features(learner, data)
                end
                @logged_testset $FEATURES verbosity begin
                    LearnAPI.features(learner, observations)
                end
            end

            if !(isnothing(X))
                @logged_testset $OBS_INVOLUTIVITY verbosity begin
                    _observations = LearnAPI.obs(model, X)
                    Test.@test LearnAPI.obs(model, _observations) == _observations
                end
            end

            model2 = @logged_testset $SERIALIZATION verbosity begin
                small_model = LearnAPI.strip(model)
                io = IOBuffer()
                Serialization.serialize(io, small_model)
                seekstart(io)
                Serialization.deserialize(io)
            end

            if :(LearnAPI.predict) in _functions
                # get data argument for `predict`:
                if isnothing(X)
                    args = ()
                    message = $PREDICT_HAS_NO_FEATURES
                else
                    args = (X,)
                    message = $PREDICT_HAS_FEATURES
                end
                yhat = @logged_testset message verbosity begin
                    [LearnAPI.predict(model, k, args...) for k in _kinds_of_proxy]
                    LearnAPI.predict(model, args...)
                end

                @logged_testset $STRIP verbosity begin
                    Test.@test @nearly(
                    LearnAPI.predict(LearnAPI.strip(model), args...) == yhat,
                    )
                end

                @logged_testset $STRIP2 verbosity begin
                    Test.@test @nearly LearnAPI.predict(model2, args...) == yhat
                end

                if !isnothing(X)
                    @logged_testset $OBS_AND_PREDICT verbosity begin
                        Test.@test all(_kinds_of_proxy) do kind
                            @nearly(
                                LearnAPI.predict(model, kind, LearnAPI.obs(model, X)) ==
                                    LearnAPI.predict(model, kind, X),
                            )
                        end
                    end
                end
            end

            if :(LearnAPI.transform) in _functions
                # get data argument for `transform`:
                if isnothing(X)
                    args = ()
                    message = $TRANSFORM_HAS_NO_FEATURES
                else
                    args = (X,)
                    message = $TRANSFORM_HAS_FEATURES
                end
                W = @logged_testset message verbosity begin
                    LearnAPI.transform(model, args...)
                end

                @logged_testset $STRIP_TRANSFORM verbosity begin
                    Test.@test @nearly(
                    LearnAPI.transform(LearnAPI.strip(model), args...) == W,
                    )
                end

                @logged_testset $STRIP2_TRANSFORM verbosity begin
                    Test.@test @nearly LearnAPI.transform(model2, args...) == W
                end

                if !isnothing(X)
                    @logged_testset $OBS_AND_TRANSFORM verbosity begin
                        Test.@test @nearly(
                            LearnAPI.transform(model, LearnAPI.obs(model, X)) == W,
                        )
                    end
                end
            end

            if :(LearnAPI.inverse_transform) in _functions && !isnothing(X)
                X2 = @logged_testset $INVERSE_TRANSFORM verbosity begin
                    LearnAPI.inverse_transform(model, W)
                end

                @logged_testset $STRIP_INVERSE verbosity begin
                    Test.@test @nearly(
                        LearnAPI.inverse_transform(LearnAPI.strip(model), W) == X2,
                    )
                end

                @logged_testset $STRIP2_INVERSE verbosity begin
                    Test.@test @nearly LearnAPI.inverse_transform(model2, W) == X2
                end
            end

            if !_is_static
                _observations =
                    @logged_testset $OBS_INVOLUTIVITY_FIT verbosity begin
                        _observations = LearnAPI.obs(learner, data)
                        Test.@test LearnAPI.obs(learner, _observations) == _observations
                        _observations
                    end

                @logged_testset $OBS_AND_FIT verbosity begin
                    obsmodel =
                        LearnAPI.fit(learner, _observations; verbosity=verbosity - 1)
                    if :(LearnAPI.predict) in _functions
                        Test.@test @nearly(
                            LearnAPI.predict(model, args...) ==
                                LearnAPI.predict(obsmodel, args...),
                        )
                    end
                    if :(LearnAPI.transform) in _functions
                        Test.@test @nearly(
                            LearnAPI.transform(model, args...) ==
                                LearnAPI.transform(obsmodel, args...),
                        )
                    end
                    if :(LearnAPI.inverse_transform) in _functions
                        Test.@test @nearly(
                            LearnAPI.inverse_transform(model, W) ==
                                LearnAPI.inverse_transform(obsmodel, W),
                        )
                    end
                end

                model3 =
                    @logged_testset $SELECTED_FOR_FIT verbosity begin
                        data3 = LearnTestAPI.learner_get(learner, data)
                        if _data_interface isa LearnAPI.RandomAccess
                            LearnAPI.fit(learner, data3; verbosity=verbosity-1)
                        else
                            nothing
                        end
                    end
            end

            if !isnothing(X)
                X3 = @logged_testset $SELECTED verbosity begin
                    LearnTestAPI.model_get(model, X)
                end
                if _data_interface isa LearnAPI.RandomAccess
                    if :(LearnAPI.predict) in _functions
                        if _is_static
                            @logged_testset(
                                $PREDICT_ON_SELECTIONS1,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.predict(model, X3) == yhat
                                end,
                            )
                        else
                            @logged_testset(
                                $PREDICT_ON_SELECTIONS2,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.predict(model, X3) == yhat
                                    Test.@test @nearly LearnAPI.predict(model3, X3) == yhat
                                    Test.@test @nearly LearnAPI.predict(model3, X3) == yhat
                                end,
                            )
                        end
                    end
                    if :(LearnAPI.transform) in _functions
                        if _is_static
                            @logged_testset(
                                $TRANSFORM_ON_SELECTIONS1,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.transform(model, X3) == W
                                end,
                            )
                        else
                            @logged_testset(
                                $TRANSFORM_ON_SELECTIONS2,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.transform(model, X3) == W
                                    Test.@test @nearly LearnAPI.transform(model3, X3) == W
                                    Test.@test @nearly LearnAPI.transform(model3, X3) == W
                                end,
                            )
                        end
                    end
                elseif _data_interface isa LearnAPI.FiniteIterable
                    if :(LearnAPI.predict) in _functions
                        if _is_static
                            @logged_testset(
                                $PREDICT_ON_SELECTIONS1,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.predict(model, X3) == yhat
                                end,
                            )
                        else
                            @logged_testset(
                                $PREDICT_ON_SELECTIONS2,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.predict(model, X3) == yhat
                                    Test.@test @nearly LearnAPI.predict(model3, X3) == yhat
                                    Test.@test @nearly LearnAPI.predict(model3, X3) == yhat
                                end,
                            )
                        end
                    end
                    if :(LearnAPI.transform) in _functions
                        if _is_static
                            @logged_testset(
                                $TRANSFORM_ON_SELECTIONS1,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.transform(model, X3) == W
                                end,
                            )
                        else
                            @logged_testset(
                                $TRANSFORM_ON_SELECTIONS2,
                                verbosity,
                                begin
                                    Test.@test @nearly LearnAPI.transform(model, X3) == W
                                    Test.@test @nearly LearnAPI.transform(model3, X3) == W
                                    Test.@test @nearly LearnAPI.transform(model3, X3) == W
                                end,
                            )
                        end
                    end
                end
            end

            # target

            if :(LearnAPI.target) in _functions
                _y = @logged_testset $TARGET0 verbosity begin
                    LearnAPI.target(learner, data)
                end
                @logged_testset $TARGET verbosity begin
                    LearnAPI.target(learner, observations)
                end
                @logged_testset $TARGET_SELECTIONS verbosity begin
                    LearnTestAPI.learner_get(
                        learner,
                        data,
                        data->LearnAPI.target(learner, data),
                    )
                    LearnTestAPI.learner_get(
                        learner,
                        observations,
                        data->LearnAPI.target(learner, data),
                    )
                end
            end

            # weights

            _w = @logged_testset $WEIGHTS verbosity begin
                LearnAPI.weights(learner, observations)
            end

            if !(isnothing(_w))
                @logged_testset $WEIGHTS_IN_FUNCTIONS verbosity begin
                    Test.@test :(LearnAPI.weights) in _functions
                end
                w = @logged_testset $WEIGHTS_SELECTIONS verbosity begin
                    LearnTestAPI.learner_get(
                        learner,
                        data,
                        data->LearnAPI.weights(learner, data),
                    )
                end
            else
                @logged_testset $WEIGHTS_NOT_IN_FUNCTIONS verbosity begin
                    Test.@test !(:(LearnAPI.weights) in _functions)
                end
            end

            # update

            if :(LearnAPI.update) in _functions
                _is_static && throw($ERR_STATIC_UPDATE)
                @logged_testset $UPDATE verbosity begin
                    LearnAPI.update(model, data; verbosity=0)
                end
                # only test hyperparameter replacement in case of iteration parameter:
                iter = LearnAPI.iteration_parameter(learner)
                if !isnothing(iter)
                    @logged_testset $UPDATE_ITERATIONS verbosity begin
                        n = getproperty(learner, iter)
                        newmodel = LearnAPI.update(model, data, iter=>n+1; verbosity=0)
                        newlearner = LearnAPI.clone(learner, iter=>n+1)
                        Test.@test LearnAPI.learner(newmodel) == newlearner
                        abinitiomodel = LearnAPI.fit(newlearner, data; verbosity=0)
                        # prepare to test `predict` and `transform` are the same for ab
                        # initio model and for updated model:
                        if isnothing(X)
                            args = ()
                        else
                            args = (X,)
                        end
                        operations = intersect(
                            _functions,
                            [:(LearnAPI.predict), :(LearnAPI.transform)],
                        )
                        for ex in operations
                            op = eval(:($ex))
                            left = op(newmodel, args...)
                            right = op(newmodel, args...)
                            agree = @nearly left == right
                            if agree
                                Test.@test true
                            else
                                @error $ERR_BAD_UPDATE
                            end
                        end
                    end
                end
            end

            # accessor functions
            # `learner`, `extras`, `strip`, `already` tested above

            if :(LearnAPI.coefficients) in _functions
                @logged_testset $COEFFICIENTS verbosity begin
                    coefs = LearnAPI.coefficients(model)
                    Test.@test coefs isa AbstractVector{
                        <:Pair{<:Any,<:Union{Real,AbstractVector{<:Real}}}
                    }
                    if :(LearnAPI.feature_names) in _functions
                        Test.@test Set(LearnAPI.feature_names(model)) ==
                            Set(first.(coefs))
                    end
                end
            end

            if :(LearnAPI.intercept) in _functions
                @logged_testset $INTERCEPT verbosity begin
                    c = LearnAPI.intercept(model)
                    Test.@test c isa Union{Real,AbstractVector{<:Real}}
                end
            end

            if :(LearnAPI.feature_importances) in _functions
                @logged_testset $FEATURE_IMPORTANCES verbosity begin
                    fis = LearnAPI.feature_importances(model)
                    Test.@test fis isa AbstractVector{
                        <:Pair{<:Any,<:Real}
                    }
                    if :(LearnAPI.feature_names) in _functions
                        Test.@test Set(LearnAPI.feature_names(model)) ==
                            Set(first.(fis))
                    end
                end
            end

            if :(LearnAPI.training_losses) in _functions
                @logged_testset $TRAINING_LOSSES verbosity begin
                    losses = LearnAPI.training_losses(model)
                    Test.@test losses isa AbstractVector
                end
            end

            if :(LearnAPI.out_of_sample_losses) in _functions
                @logged_testset $OUT_OF_SAMPLE_LOSSES verbosity begin
                    losses = LearnAPI.out_of_sample_losses(model)
                    Test.@test losses isa AbstractVector
                end
            end

            if :(LearnAPI.training_scores) in _functions
                @logged_testset $TRAINING_SCORES verbosity begin
                    losses = LearnAPI.training_scores(model)
                    Test.@test losses isa AbstractVector
                end
            end

            if :(LearnAPI.predictions) in _functions
                _is_static && throw(ERR_STATIC_PREDICTIONS)
                @logged_testset $PREDICTIONS verbosity begin
                    Test.@test @nearly(
                        LearnAPI.predictions(model) == LearnAPI.predict(model, X),
                    )
                end
            end

            if :(LearnAPI.out_of_sample_indices) in _functions
                :(LearnAPI.predictions) in _functions || throw(ERR_OOS_INDICES)
                @logged_testset $OUT_OF_SAMPLE_INDICES verbosity begin
                    Test.@test(
                        LearnAPI.out_of_sample_indices(model) isa AbstractVector{<:Integer}
                    )
                end
            end

            # traits
            # `constructor`,  `functions`, `kinds_of_proxy`, `tags`, `nonlearners`,
            # `iteration_parameter`, `data_interface`, `is_static` tested already above

            @logged_testset $PKG_NAME verbosity begin
                pkg_name = LearnAPI.pkg_name(learner)
                Test.@test pkg_name isa String
                if pkg_name == "unknown"
                    push!(missing_traits, :(LearnAPI.pkg_name))
                else
                    Test.@test !occursin(".jl", pkg_name)
                end
            end

            @logged_testset $PKG_LICENSE verbosity begin
                pkg_license = LearnAPI.pkg_license(learner)
                pkg_license == "unknown" && push!(missing_traits, :(LearnAPI.pkg_license))
                Test.@test pkg_license isa String
            end

            @logged_testset $DOC_URL verbosity begin
                doc_url = LearnAPI.doc_url(learner)
                if doc_url == "unknown"
                    push!(missing_traits, :(LearnAPI.doc_url))
                else
                    Test.@test IsURL.isurl(doc_url)
                end
            end

            @logged_testset $LOAD_PATH verbosity begin
                path = LearnAPI.load_path(learner)
                if path == "unknown"
                    push!(missing_traits, :(LearnAPI.pkg_name))
                else
                    pkg_name = split(path, ".") |> first
                    name = LearnAPI.pkg_name(learner)
                    name == "unknown" || Test.@test name == pkg_name
                    command = "import "*path
                    ex = Meta.parse(command)
                    eval(ex)
                end
            end

            @logged_testset $HUMAN_NAME verbosity begin
                Test.@test _human_name isa String
            end

            S = LearnAPI.fit_scitype(learner)
            if S == Union{}
                push!(missing_traits, :(LearnAPI.fit_scitype))
            else
                @logged_testset $FIT_SCITYPE verbosity begin
                    Test.@test ScientificTypes.scitype(data) <: S
                end
            end

            S = LearnAPI.target_observation_scitype(learner)
            testable = :(LearnAPI.target) in _functions &&
               _data_interface in (LearnAPI.RandomAccess(), LearnAPI.FiniteIterable())
            if S == Any && (LearnAPI.target) in _functions
                push!(missing_traits, :(LearnAPI.target_observation_scitype))
            elseif testable
                @logged_testset $TARGET_OBSERVATION_SCITYPE verbosity begin
                    Test.@test all([o for o in _y]) do o
                        ScientificTypes.scitype(o) <: S
                    end
                end
            end

        end # for loop over datasets
        verbosity > 0 && !isempty(missing_traits) &&
            @info $MISSING_TRAITS(missing_traits, _human_name)
        verbosity > 0 && @info "------ @testapi - $_human_name - tests complete ------"
        nothing
    end # quote
end # macro
