using Documenter
using LearnTestAPI

const  REPO = Remotes.GitHub("JuliaAI", "LearnTestAPI.jl")

makedocs(
    modules=[LearnTestAPI,],
    format=Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        collapselevel = 1,
    ),
    pages=[
        "Home" => "index.md",
    ],
    sitename="LearnTestAPI.jl",
    warnonly = [:cross_references, :missing_docs],
    repo = Remotes.GitHub("JuliaAI", "LearnTestAPI.jl"),
)

deploydocs(
    devbranch="dev",
    push_preview=false,
    repo="github.com/JuliaAI/LearnTestAPI.jl.git",
)
