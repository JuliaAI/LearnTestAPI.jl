```@raw html
<script async defer src="https://buttons.github.io/buttons.js"></script>
<span style="color: #9558B2;font-size:4.5em;">
LearnTestAPI.jl</span>
<br>
<span style="color: #9558B2;font-size:1.6em;font-style:italic;">
Tool for testing implementations of
<a href=https://juliaai.github.io/LearnAPI.jl/dev/ style="text-decoration: underline">LearnAPI.jl</a></span>
<br>
<br>
```

## Quick start

```@docs
LearnTestAPI
```

LearnAPI.jl and LearnTestAPI.jl have synchronized releases. For example, LearnTestAPI.jl
version 0.2.3 will generally support all LearnAPI.jl versions 0.2.*.

!!! warning

    New releases of LearnTestAPI.jl may add tests to `@testapi`, and this may result in
    new failures in client package test suites. Nevertheless, adding a test to `@testapi`
    is not considered a breaking change to LearnTestAPI, unless the addition supports a
    breaking release of LearnAPI.jl.


## The @testapi macro

```@docs
LearnTestAPI.@testapi
```

## Learners for testing

LearnTestAPI.jl provides some simple, tested, LearnAPI.jl implementations, which may be
useful for testing learner wrappers and meta-algorithms. 

```@docs
LearnTestAPI.Ridge
LearnTestAPI.BabyRidge
LearnTestAPI.TruncatedSVD
LearnTestAPI.Selector
LearnTestAPI.FancySelector
LearnTestAPI.NormalEstimator
LearnTestAPI.Ensemble
```


## Private methods

For LearnTestAPI.jl developers only, and subject to breaking changes:

```@docs
LearnTestAPI.@logged_testset
LearnTestAPI.@nearly
LearnTestAPI.isnear
LearnTestAPI.learner_get
LearnTestAPI.model_get
LearnTestAPI.verb
LearnTestAPI.filter_out_verbosity
```
