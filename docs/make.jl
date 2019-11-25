using AxisArrays
using Distributions
using Documenter
using IndexedDistributions
using Metrics

makedocs(;
    modules=[Metrics],
    format=Documenter.HTML(
        assets=[
            "assets/invenia.css",
            "assets/logo.png",
        ],
        prettyurls=false,
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => "examples.md"
    ],
    repo="https://gitlab.invenia.ca/invenia/Metrics.jl/blob/{commit}{path}#L{line}",
    sitename="Metrics.jl",
    authors="Invenia Technical Computing Corporation",
    strict=true,
    checkdocs=:none,
)
