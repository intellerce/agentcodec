# Commercial licensing

AgentCodec is released under the **PolyForm Noncommercial License 1.0.0**
(see [LICENSE](LICENSE)). That license permits any noncommercial use —
research, teaching, personal experimentation, internal evaluation at a
non-profit or educational institution, and similar — without payment or
notification.

If your intended use is **commercial**, you need a separate license from
the authors. Examples that require a commercial license include:

- Selling the software, modified or unmodified, as part of a product.
- Running the software inside an internal pipeline that supports a
  for-profit business or a paid service.
- Offering AgentCodec, or a derivative, as a hosted API or SaaS.
- Bundling AgentCodec inside another commercial library or framework.

If you're unsure whether your use is commercial, the PolyForm FAQ at
<https://polyformproject.org/about/faq/> gives helpful examples. When in
doubt, reach out and ask.

## How to inquire

Email: **research@intellerce.com**

Please include in your message:

1. Your organization and a one-paragraph description of the intended use.
2. Approximate deployment scale (per-month request volume, number of
   internal users, geographic scope).
3. Whether you need a self-hosted SemKNN backend (private profiles + the
   trained q-matrix) or only a commercial license for the public client.
4. Any deadline you're working against.

We aim to respond within two business days. Standard commercial license
terms are available; bespoke arrangements for research consortia,
educational software vendors, and high-volume deployments are negotiable.

## What a commercial license covers

A commercial license grants the same rights as the public PolyForm-NC
license, plus:

- The right to use the software for any commercial purpose.
- Optional access to the private SemKNN backend image and the trained
  profile artifacts, for on-premise or air-gapped deployments.
- A direct support channel and a defined turnaround for security
  notifications.

It does **not** affect the public release: the open codebase remains
PolyForm-NC for everyone.

## A note on self-hosting SemKNN

The public release contains the SemKNN backend's *service code*, but
the service is useless without the trained artifacts — the SemKNN
q-matrix and the training-set embeddings produced from our paid
benchmark runs. Those artifacts are not redistributed with this
release. To self-host SemKNN you need either:

- an **academic / research license**, which we grant case-by-case at
  no cost for non-commercial research use (publication, teaching,
  on-prem evaluation at a non-profit institution), or
- a **commercial license**, with terms tailored to your deployment.

In both cases, email the address above and tell us about the intended
use. The shipped artifacts include a versioned profile bundle and a
verification key so you can confirm you're running the artifacts that
match a given paper's reported numbers.
