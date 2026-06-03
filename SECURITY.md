# Security policy

AgentCodec runs LLM workloads and ships an anonymous-telemetry path.
Both surfaces have failure modes that affect users' data, money, and
trust. We treat them seriously and we want to hear from you.

## Reporting a vulnerability

**Please do not file public issues for security or privacy problems.**

The preferred channel is **GitHub private vulnerability reporting**: on
the public repo, open the **Security** tab and click **"Report a
vulnerability"**. This opens a private advisory only you and the
maintainers can see, lets us collaborate (and test a fix on a private
fork) without tipping off attackers, and handles CVE issuance and
reporter credit on disclosure.

If you don't have a GitHub account or prefer email, write to
**research@intellerce.com** instead. Either way, include:

- A description of the issue and the impact you observe (or expect).
- A minimal reproducer if you can — Python snippet + YAML config.
- The AgentCodec version
  (`python -c "import agentcodec; print(agentcodec.__version__)"`).
- Whether the issue is in the published library, the SemKNN backend, or
  the telemetry collector.

We aim to acknowledge within **two business days** and to ship a fix
or a written plan within **fourteen calendar days** of confirming the
report. We will credit the reporter in the changelog and the release
notes unless asked otherwise.

## In scope

- Anything that leaks prompt text, model output, reference answers,
  API keys, or user identifiers through:
  - The remote SemKNN call payload (`/route` body).
  - The anonymous telemetry payload (`/telemetry` body).
  - The `ReliabilityResult` trace (e.g. an accidental field with raw
    text under an "innocent" key name).
- Anything that causes the library to make HTTP calls the user did not
  authorize (e.g. an outbound request to an unexpected host).
- Anything that disables `AGENTCODEC_TELEMETRY=0` or
  `telemetry.enabled: false` — those are contractual kill switches.
- Anything that bypasses the schema-pin denylist in
  `agentcodec.telemetry._scrub()` (the privacy fuse).
- Cost-accounting bugs that lead to wildly wrong `cost_usd` without
  the `default_fallback` warning firing — financial-impact bugs.
- Code execution / deserialization issues in the YAML / JSON loaders.

## Out of scope

- Reports about the underlying LLM provider's API behaving badly (rate
  limits, content policy, hallucinations). Report those upstream.
- Vulnerabilities in `pip install`-ed dependencies — those go to the
  upstream project.
- DoS via "I sent a 10 MB prompt and the request was slow."

## Known caveats users should plan around

We document these so they're not surprises:

- **Cancellation does not abort in-flight HTTP requests.** The library
  uses worker threads for sync LLM SDK calls; cancelling the consumer
  drops the response but does not cancel server-side work. You may be
  billed for tokens generated after cancel.
- **Sentence embeddings are lossy but not provably zero-leakage.** The
  prompt-embedding the SemKNN router sends to the backend is a
  unit-norm BGE vector. Research-grade inversion attacks (`vec2text`)
  can sometimes recover topic-level content. The `bge-small` default
  reduces this. If your environment requires zero embedding leakage,
  set `AGENTCODEC_TELEMETRY=0` and avoid the SemKNN router (use
  `fixed` or `acm_table`).
- **The `default_fallback` cost tier is a stub.** If a model's pricing
  isn't in `MODEL_COSTS` and the user didn't set `cost_per_1m`, the
  library uses $2/$8 and logs a loud warning. This is intentional but
  documented as a gotcha.

## Supported versions

For security purposes, only the **latest released version** is
supported. There are no LTS branches at this stage. If you're using an
older version, the upgrade path is usually trivial; if not, file an
issue and we'll help.

## Encryption

GitHub private vulnerability reporting is already a confidential
channel, so no extra encryption is needed when you use it.

For email reports, `research@intellerce.com` accepts PGP if you prefer.
We will publish the public key on the repo and the website before the
first public push; until then, please send a short email asking for it.
