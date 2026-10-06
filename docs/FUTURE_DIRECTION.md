# Future direction

> **Nothing on this page exists yet.** It is a sketch of where the antenna
> builder is intended to go, published so you can judge whether the tool is
> heading somewhere useful to you. None of it is implemented, none of it is
> promised, and none of it should factor into a decision about what the Studio
> can do today. For that, read
> [Current limitations](LIMITATIONS.md).

## Where things stand

Today a design **must** start from one of three validated recipes — an inset-fed
rectangular patch, a probe-fed circular patch, or a centre-fed dipole. The recipe
is the construction capability: it decides which antenna families are reachable
at all. Compositional editing then works on top of that seed, so you can cut a
slot into a patch, but you cannot arrive at an antenna the recipes do not
already describe.

That design made the first version trustworthy. It also makes the recipe list the
ceiling.

## The intended inversion

The direction is to **invert that relationship**: recipes become optional
starting templates, and a library of reusable CAD and electromagnetic tools
becomes the real construction capability boundary.

| | Today | Intended |
| --- | --- | --- |
| What bounds the possible designs | The three recipes | The registered tool library |
| Role of a recipe | Mandatory seed | Optional convenience template |
| Adding an antenna family | Write a new recipe | Often none needed, if the tools already compose |
| Starting from nothing | Not available | A supported path |

Illustratively: instead of `recipe.select("inset_patch_v2")` being the only way
to obtain a patch, a planner would compose a substrate, a ground plane, a
radiating conductor and a feed from general tools — and the existing inset-patch
recipe would survive as a shortcut that produces the same validated result in one
step. A Yagi-Uda, a slot antenna or a stacked patch would then become reachable
through composition rather than through a new recipe, provided the necessary
primitives and the engineering checks that police them exist.

## What would have to be true first

This is the honest part, and it is why the page is a sketch rather than a plan.
Making tools the boundary is only an improvement if the guardrails scale with
the freedom:

- **The primitive set has to be genuinely general** — arbitrary extrusions,
  revolutions, true 3D Booleans and curved surfaces, not the planar-XY
  extrusion evaluator in place today.
- **Validation has to cover geometry nobody wrote a recipe for.** Current
  checks lean on knowing what shape to expect. Conductor connectivity, minimum
  copper widths, port attachment and feed sanity would all need to hold for
  shapes that were never anticipated.
- **Analytical grounding has to come from somewhere.** Recipes supply sensible
  starting dimensions. A freely composed antenna has no analytical baseline, so
  either the tool library supplies one or the first simulation becomes the only
  feedback.
- **Refusal has to stay honest.** The value of the current design is that
  unsupported requests fail cleanly instead of producing confident nonsense.
  Broadening what is reachable must not broaden what is silently wrong.

Getting those right matters more than getting there quickly. Adding antenna
names before the guardrails exist would make the feature look broader while
making it less trustworthy — the opposite of the point.

## What will not change

Whatever the construction boundary becomes:

- The language model will keep planning only through registered, schema-checked
  tools. It will not write CAD or solver code.
- A deterministic executor will keep owning geometry, validation and export.
- Generated geometry will remain a **starting point for you to simulate and
  validate**, never a substituted EM result.
