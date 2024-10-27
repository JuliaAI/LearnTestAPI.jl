# LearnTestAPI.jl

Developers' tool for testing implementations of the LearnAPI.jl interface for alogorithms
in machine learning and statstics

[![Lifecycle:Maturing](https://img.shields.io/badge/Lifecycle-Maturing-007EC6)](ROADMAP.md)
[![Build Status](https://github.com/JuliaAI/LearnTestAPI.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/LearnTestAPI.jl/actions)
[![codecov](https://codecov.io/gh/JuliaAI/LearnTestAPI.jl/graph/badge.svg?token=9IWT9KYINZ)](https://codecov.io/gh/JuliaAI/LearnTestAPI.jl?branch=dev)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaai.github.io/LearnTestAPI.jl/dev/)

# Quick start

If your package defines an object `learner` implementing the [LearnAPI.jl]() interface,
then put something like this in your test suite:

```julia
using LearnTestAPI

# create some test data:
X = (
    feautre1 = [1, 2, 3],
    feature2 = ["a", "b", "c"],
    feature3 = [10.0, 20.0, 30.0],
)
y = [1, 2, 3]
data = (X, y)

# bump verbosity to debug:
@testapi learner data verbosity=1
```

Once tests pass, you can set `verbosity=0` to suppress the detailed logging. 
