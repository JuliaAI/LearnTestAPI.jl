ERR_UNSUPPORTED_KWARG(arg) = ArgumentError(
    "Got unsupported keyword argument `$arg`. "
)

const LOUD = "- specify `verbosity=0` for silent testing. ------"

const QUIET = "- specify `verbosity=1` if debugging"

const CONSTRUCTOR = """

    Testing that learner can be reconstructed from its constructors.
    [Reference](https://juliaai.github.io/LearnAPI.jl/dev/reference/#learners).

  """
const NONLEARNERS = """

    Testing all elements of `LearnAPI.nonlearners(learner)` are indeed non-learners, and
    that all elements of  `LearnAPI.learners(learner)` are learners.
    [Reference](https://juliaai.github.io/LearnAPI.jl/dev/reference/#learners).

  """
const WARN_DOCUMENTATION = """

    The learner constructor is undocumented.
    [Reference](https://juliaai.github.io/LearnAPI.jl/dev/reference/#Documentation).

  """
const OBLIGATORY_FUNCTIONS =
    join(map(ex->"`:$ex`", LearnAPI.functions()[1:5]), ", ", " and ")
const FUNCTIONS = """

    Testing that `LearnAPI.functions(learner)` includes $OBLIGATORY_FUNCTIONS.

  """
const ERR_MISSINNG_OBLIGATORIES =
    "These obligatory functions are missing from the return value of "*
    "`LearnAPI.functions(learner)`; "

const FUNCTIONS3 = """

    Testing that `LearnAPI.functions(learner)` includes `:(LearnAPI.features).`

  """
const FUNCTIONS4 = """

    Testing that `LearnAPI.functions(learner)` exludes `:(LearnAPI.features)`, as
    `LearnAPI.is_static(learner)` is `true`.

  """
const TAGS = """

    Testing that `LearnAPI.tags(learner)` has correct form. List allowed tags with
    `LearnAPII.tags()`.

  """
const KINDS_OF_PROXY = """

    Testing that `LearnAPI.kinds_of_proxy(learner)` is overloaded and has valid form.
    [Reference](https://juliaai.github.io/LearnAPI.jl/dev/kinds_of_target_proxy/). List
    allowed kinds of proxy with `LearnAPI.kinds_of_proxy()`.

  """
const FIT_IS_STATIC = """

    `LearnAPI.is_static(learner)` is `true`. Therefore attempting to call
    `fit(learner)`.

  """
const FIT_IS_NOT_STATIC = """

    Attempting to call `fit(learner, data)`. If you implemented `fit(learner)` instead,
    then you need to arrange `LearnAPI.is_static(learner) == true`.

  """
const LEARNER = """

    Attempting to call `LearnAPI.strip(model)` and to check that applying
    `LearnAPI.learner` to the result or to `model` returns `learner` in both cases.

  """
const FUNCTIONS2 = """

    Checking for overloaded LearnAPI.jl functions missing from the return value of
    `LearnAPI.functions(learner)`. Run `??LearnAPI.functions` for a
    checklist.

  """
const ERR_MISSING_FUNCTIONS =
    "The following overloaded functions are missing from the return value of"*
    "`LearnAPI.functions(learner)`: "

const OBS = """

    Attempting to call `observations = obs(learner, data)`.

  """
const FEATURES0 = """

    Attempting to call `LearnAPI.features(learner, data)`.

  """
const FEATURES = """

    Attempting to call `LearnAPI.features(learner, observations)`.

  """
const OBS_INVOLUTIVITY = """

    Testing that `obs(model, obs(model, data)) == obs(model, data)` (involutivity).

  """
const SERIALIZATION = """

    Checking that `LearnAPI.strip(model)` can be Julia-serialized/deserialized.

  """
const PREDICT_HAS_NO_FEATURES = """

    Attempting to call `predict(model, kind)` for each `kind` in
    `LearnAPI.kinds_of_proxy(learner)`. (We are not providing `predict` with a data
    argument because `features(obs(learner, data)) == nothing`).


  """
const PREDICT_HAS_FEATURES = """

    Attempting to call `predict(model, kind, X)` for each `kind` in
    `LearnAPI.kinds_of_proxy(learner)`. Here `X = LearnAPI.features(obs(learner, data))`.

  """
const STRIP = """

    Checking that `predict(model, ...)` gives the same answer if we replace `model` with
    `LearnAPI.strip(model)`.

  """
const STRIP2 = """

    Checking that `predict(model, X)` can be called on the deserialized version of `model`
    without a change in return value.

  """
const OBS_AND_PREDICT = """

    Testing that replacing `X` in `predict(learner, kind, X)` with `obs(model, X)` gives
    the same output, for each supported `kind` of proxy.

  """
const TRANSFORM_HAS_NO_FEATURES = """

    Attempting to call `transform(model)` (and not `transform(model, X)`, because
    `features(obs(learner, data)) == nothing`).

  """
const TRANSFORM_HAS_FEATURES = """

    Attempting to call `transform(model, X)`.

  """
const STRIP_TRANSFORM = """

    Testing that `transform(model, X)` gives the same answer if we replace `model` with
    `LearnAPI.strip(model)`.

  """
const STRIP2_TRANSFORM = """

    Testing  that `transform(model, X)` can be called using the deserialized version of
    `model` without a change in return value.

  """
const OBS_AND_TRANSFORM = """

    Testing that replacing `X` in `transform(learner, X)` with `obs(model, X)` gives the
    same output.

  """
const INVERSE_TRANSFORM = """

    Testing that `inverse_transform(model, W)` executes, where `W` is the output of
    `transform(model, X)` and `X = LearnAPI.features(learner, observations)`.

  """
const STRIP_INVERSE = """

    Testing that `inverse_transform(model, X)` gives the same answer if we replace
    `model` with `LearnAPI.strip(model)`.

  """
const STRIP2_INVERSE = """

    Testing that `inverse_transform(model, X)` can be called using the deserialized
    version of `model` without a change in return value.

  """
const OBS_INVOLUTIVITY_FIT =
    """

    Testing that `obs(learner, obs(learner, data)) == obs(learner, data)` (involutivity).

  """
const OBS_AND_FIT = """

    Testing that replacing `data` with `obs(learner, data)` in the call `fit(learner,
    data)` makes no difference to `predict`, `transform` or `inverse_transform` outcomes
    (where implemented).

  """
const SELECTED_FOR_FIT = """

    Testing that we can select all observations from `_data = LearnAPI.obs(learner,
    data)`, using the interface declared by `LearnAPI.data_interface(learner)`. For
    example, if this interface is `LearnAPI.RandomAccess()`, the attempted selection is
    `data3 = MLUtils.getobs(_data, 1:MLUtils.numobs(_data))`. For this interface, also
    testing that we can call `fit` on the selection, obtaining a new `fit` result
    `model3`.

  """
const SELECTED = """

    Testing that we can select all observations from `_X = LearnAPI.obs(model, X)`,
    using the interface declared by `LearnAPI.data_interface(learner)`. For example, if
    this interface is `LearnAPI.RandomAccess()`, the attempted selection is
    `X3 = MLUtils.getobs(_X, 1:MLUtils.numobs(_X))`.

  """
const PREDICT_ON_SELECTIONS1 = """

    Testing that `predict(model, X)` is unchanged by replacing `X` with `X3`
    ("subsampled" features). If not described in the log, re-run `@testapi` with
    `verbosity=1` for further explanation of `X3`.

  """
const TRANSFORM_ON_SELECTIONS1 = """

    Testing that `transform(model, X)` is unchanged by replacing `X` with `X3`
    ("subsampled" features). If not described in the log, re-run `@testapi` with
    `verbosity=1` for further  explanation of `X3`.

  """
const PREDICT_ON_SELECTIONS2 = """

    Testing that `predict(model, X)` is unchanged by replacing `model` with `model3`
    (trained on "subsampled" `fit` data) and/or replacing `X` with `X3`
    ("subsampled" features). If not described in the log, re-run `@testapi` with
    `verbosity=1` for further explanation of `model3` and `X3`.

  """
const TRANSFORM_ON_SELECTIONS2 = """

    Testing that `transform(model, X)` is unchanged by replacing `model` with `model3`
    (trained on "subsampled" `fit` data) and/or replacing `X` with `X3`
    ("subsampled" features). If not described in the log, re-run `@testapi` with
    `verbosity=1` for further explanation of `model3` and `X3`.

  """
const TARGET0 = """

    Attempting to call `LearnAPI.target(learner, data)` (fallback returns
    `last(data)`).

  """
const TARGET = """

    Attempting to call `LearnAPI.target(learner, observations)` (fallback returns
    `last(observations)`).

  """
const TARGET_SELECTIONS = """

    Checking that all observations can be extracted from `LearnAPI.target(learner, data)`
    using the data interface declared by `LearnAPI.data_interface(learner)`. Doing the
    same with `data` replaced with `observations`.

  """
const WEIGHTS = """

    Attempting to call `LearnAPI.weights(learner, observations)` (fallback returns
    `nothing`).

  """
const WEIGHTS_IN_FUNCTIONS = """

    Checking that `:(LearnAPI.weights)` is included in `LearnAPI.functions(learner)`.

  """
const WEIGHTS_SELECTIONS = """

    Checking that all observations can be extracted from `LearnAPI.weights(learner,
    observations)` using the data interface declared by
    `LearnAPI.data_interface(learner)`.

  """
const WEIGHTS_NOT_IN_FUNCTIONS = """

    Checking that `:(LearnAPI.weights)` is excluded from `LearnAPI.functions(learner)`, as
    `LearnAPI.weights` has not been overloaded.

  """
const COEFFICIENTS = """

    Checking `LearnAPI.coefficients(model)` has the correct form, and that the feature
    names match those returned by `LearnAPI.feature_names(model)`, if the latter is
    implemented.

  """
const UPDATE = """

    Calling `LearnAPI.update(model, data; verbosity=0)`.

  """
const ERR_STATIC_UPDATE = ErrorException(
    "`(LearnAPI.update)` is in `LearnAPI.functions(learner)` but "*
        "`LearnAPI.is_static(learner)` is `true`. You cannot implement `update` "*
        "for static learners. "
)
const UPDATE_ITERATIONS = """

    Attempting to increase number of iterations by one with an `update` call, giving a new
    model `newmodel`. Checking that `LearnAPI.learner(newmodel)` reflects the iteration
    update. Checking that retraining from scratch with the extra iteration gives,
    approximately, the same predictions/transformations.

  """
const ERR_BAD_UPDATE = ErrorException(
"Updating iterations by one and instead retraining from scratch are giving "*
    "different outcomes. "
)
const INTERCEPT = """

    Checking `LearnAPI.intercept(model)` has the correct form.

  """
const FEATURE_IMPORTANCES = """

    Checking `LearnAPI.feature_importances(model)` has the correct form, and that the
    feature names match those returned by `LearnAPI.feature_names(model)`, if the
    latter is implemented.

  """
const TRAINING_LOSSES = """

    Checking that `LearnAPI.training_losses(model)` is a vector.

  """
const OUT_OF_SAMPLE_LOSSES = """

    Checking that `LearnAPI.out_of_sample_losses(model)` is a vector.

  """
const TRAINING_SCORES = """

    Checking that `LearnAPI.training_scores(model)` is a vector.

  """
const PREDICTIONS = """

    Checking that `LearnAPI.predictions(model)` and `predict(model, X)` are approximately
    equal.

  """
const ERR_STATIC_PREDICTIONS = ErrorException(
    "`LearnAPI.functions(learner)` includes `:(LearnAPI.predictions)` but you may "*
    "not implement `LearnAPI.predictions` for static learners. "
)

const OUT_OF_SAMPLE_INDICES = """

    Checking that `LearnAPI.out_of_sample_indices(model)` is a vector of integers.

  """
const ERR_OOS_INDICES = ErrorException(
    "`:(LearnAPI.out_of_sample_indices)` is in the return value of "*
        "`LearnAPI.functions(learner)` but `:(LearnAPI.predictions)` is not."
)

const PKG_NAME = """

    Checking that `LearnAPI.pkg_name(learner)` is a string not including ".jl".

  """
const PKG_LICENSE = """

    Checking that `LearnAPI.pkg_license(learner)` is a string.

"""
const DOC_URL = """

    Checking that `LearnAPI.doc_url(learner)`, if overloaded, is a syntactically valid
    URL. Its reachability is *not* checked.

  """
const LOAD_PATH = """

    Checking `path = LearnAPI.load_path(learner)`, if overloaded, is a string. If
    `LearnAPI.pkg_name(learner)` is overloaded, checking this is the first part of `path`.
    Checking `:(import \$path)` can be evaluated.

  """
const HUMAN_NAME = """

    Checking that `LearnAPI.human_name(learner)` is a string.

  """
const FIT_SCITYPE = """

    Checking that `data` supplied for testing satisfies the constraints articulated by
    `LearnAPI.fit_scitype(learner)`.

  """
const TARGET_OBSERVATION_SCITYPE = """

    Checking that `LearnAPI.target(learner, observatoins)` satisifies the constraints
    articulated by `LearnAPI.target_observation_scitype(learner)`.

"""
function MISSING_TRAITS(missing_traits, name)
    list = join( map(ex->"`$ex`",collect(missing_traits)), ", ", " and ")
    return """

    The following traits, deemed optional for $name, have not been overloaded:

    $list.

  """
end
