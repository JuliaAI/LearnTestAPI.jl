# LearnTestAPI.jl

Tool for testing implementations of the
[LearnAPI.jl](https://juliaai.github.io/LearnAPI.jl/dev/) interface, for algorithms in
machine learning and statisticsp

[![Build Status](https://github.com/JuliaAI/LearnTestAPI.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/LearnTestAPI.jl/actions)
[![codecov](https://codecov.io/gh/JuliaAI/LearnTestAPI.jl/graph/badge.svg?token=9IWT9KYINZ)](https://codecov.io/gh/JuliaAI/LearnTestAPI.jl?branch=dev)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaai.github.io/LearnTestAPI.jl/dev/)

# Code snippet

```julia
using LearnTestAPI
using StableRNGs

rng=StableRNG(123)
learner = MyRegressor(; rng)
X = rand(rng, 3, 100)
y = rand(rng, 100)
@testapi learner (X, y) verbosity=1
```

Documentation is [here](https://juliaai.github.io/LearnTestAPI.jl/dev/).
