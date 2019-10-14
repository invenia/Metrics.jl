using AxisArrays
using Distributions
using Documenter
using IndexedDistributions
using Metrics

makedocs(;
    modules=[Metrics],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://gitlab.invenia.ca/invenia/Metrics.jl/blob/{commit}{path}#L{line}",
    sitename="Metrics.jl",
    authors="Invenia Technical Computing Corporation",
    assets=[
        "assets/invenia.css",
        "assets/logo.png",
    ],
    strict=true,
    html_prettyurls=false,
    checkdocs=:none,
)
