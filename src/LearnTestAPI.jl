"""
    LearnTestAPI

Module for testing implementations of the interfacde defined in
[LearnAPI.jl](https://juliaai.github.io/LearnAPI.jl/dev/).

If your package defines an object `learner` implementing the interface, then put something
like this in your test suite:

```julia
using LearnTestAPI

# create some test data:
X = ...
y = ...
data = (X, y)

# bump verbosity to debug:
@testapi learner data verbosity=1
```

Once tests pass, set `verbosity=0` to suppress the detailed logging.

For details and options see [`LearnTestAPI.@testapi`](@ref)

"""
module LearnTestAPI

using LearnAPI
import Test
import Serialization
import MLUtils
import StableRNGs
import InteractiveUtils
import MacroTools
import IsURL

include("tools.jl")
include("logging.jl")
include("testapi.jl")
include("learners/static_algorithms.jl")
include("learners/regression.jl")
include("learners/ensembling.jl")
# next learner excluded because of heavy dependencies:
# include("learners/gradient_descent.jl")
include("learners/incremental_algorithms.jl")
include("learners/dimension_reduction.jl")

export @testapi

end # module
