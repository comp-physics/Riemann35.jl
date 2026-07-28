"""
    run_params.jl — self-maintaining provenance for parameterised probes and validators.

WHY THIS EXISTS. `probe_poiseuille.jl` reports a flow rate and prints a header carrying
`g`, cells-per-mean-free-path and the order. It does NOT print the march length or the CFL
factor, both of which are `ENV`-configurable and both of which change the answer. The
published 26-moment flow-rate table was produced at `PS_TEND=1.2` rather than the default
`0.4`; nothing in the log recorded that, so two runs differing threefold in marched time
emit log files identical in every printed setting. Reproducing the table meant discovering
the parameter by search, and the number it produced turned out to be a transient sampled at
a tenth of its settling time.

THE FIX IS RECORD-AT-READ, NOT A LONGER HEADER. A hand-written header is a second list that
has to be kept in step with the first, and it will drift again the next time a knob is
added -- every script surveyed printed some of its parameters and omitted others. Reading a
parameter through `envparam` registers it, and `print_run_header` prints exactly what was
registered. You cannot forget to print a parameter you never read, and you cannot read one
without it being printed.

USAGE, a drop-in for the `get(ENV, ...)` idiom (delete the `ENV,`):

    const CPL  = parse(Float64, envparam("PS_CPL",  "6.0"))
    const TEND = parse(Float64, envparam("PS_TEND", "0.4"))
    print_run_header("POISEUILLE FLOW RATE vs KNUDSEN NUMBER")

Overridden values are flagged, so a non-default run is visible at a glance rather than
inferred by comparing against the source.
"""

const _RUN_PARAMS = Tuple{String,String,Bool}[]

"""
    envparam(key, default) -> String

Read environment variable `key`, falling back to `default`, and register it for
`print_run_header`. Returns the raw string; parse it at the call site exactly as with
`get(ENV, key, default)`, so converting a script is a mechanical edit.

Re-reading the same key does not duplicate the entry, so this is safe in a loop or in a
script that is `include`d more than once.
"""
function envparam(key::AbstractString, default::AbstractString)
    raw = get(ENV, String(key), String(default))
    overridden = haskey(ENV, String(key))
    idx = findfirst(p -> p[1] == String(key), _RUN_PARAMS)
    if idx === nothing
        push!(_RUN_PARAMS, (String(key), raw, overridden))
    else
        _RUN_PARAMS[idx] = (String(key), raw, overridden)
    end
    raw
end

"""
    print_run_header(title; io=stdout, extra=())

Print `title` followed by every parameter registered through `envparam`, marking any that
were overridden from their default. `extra` carries derived or hard-coded quantities that
are not environment variables but still change the answer --- pass them as
`("Pr" => "2/3", "omega" => "0.81")` rather than burying them in prose.

Emitting this immediately before the results table means the log is self-describing: any
run can be reproduced from its own output, which was the property the flow-rate table
lacked.
"""
function print_run_header(title::AbstractString; io::IO = stdout, extra = ())
    line = "=" ^ 92
    println(io, line)
    println(io, title)
    if isempty(_RUN_PARAMS) && isempty(extra)
        println(io, "  (no registered parameters)")
    else
        println(io, "  run parameters -- '*' marks a value overridden from its default:")
        w = maximum(length(p[1]) for p in _RUN_PARAMS; init = 0)
        w = max(w, maximum((length(String(first(e))) for e in extra); init = 0))
        for (k, v, ov) in _RUN_PARAMS
            println(io, "    ", ov ? "* " : "  ", rpad(k, w), " = ", v)
        end
        for e in extra
            println(io, "      ", rpad(String(first(e)), w), " = ", last(e))
        end
    end
    println(io, line)
    flush(io)
    nothing
end

"Registered parameters as (key, value, overridden) triples — for tests."
run_params() = copy(_RUN_PARAMS)

"Forget all registered parameters. Only needed when one process runs several cases."
reset_run_params!() = (empty!(_RUN_PARAMS); nothing)
