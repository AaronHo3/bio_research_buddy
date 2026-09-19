# Deploying the demo

The demo is a stateless web service. Each question is answered from live PubMed,
so there is no database to provision and nothing to keep in sync. The only data
the container holds is `web/data/answers.json`, the recorded example answers.

That makes hosting simple, with one real constraint: **a question takes twelve to
thirty seconds**, because it searches PubMed, fetches full text, and calls a model
twice. Platforms that cap a request at ten seconds cannot run this, which rules
out the default serverless tiers of Vercel and Netlify. It needs a long-running
process.

## Before exposing it publicly

A public URL wired to your API key means strangers spend your money. Three things
stand between that and a surprise bill, and the first is the one that matters:

- `DEMO_DAILY_TOKENS` is a hard daily ceiling across planning and synthesis
  together. About 5,700 tokens go on a question, so the default of 200,000 is
  roughly 35 questions a day. Past the ceiling the demo answers with the
  deterministic extractive provider and says so on the page rather than failing.
- Answers are cached by normalised question, so a link that gets shared is paid
  for once rather than once per visitor.
- Requests are rate limited per address.

Set a spend limit in the Anthropic console as well. The ceiling here is
in-memory, so a host that restarts often could spend more than one day's
allowance in a day; the console limit is the backstop that does not reset.

## Render

`render.yaml` in the repository root is a blueprint, so the configuration is
reviewable rather than clicked into a dashboard.

1. Push the repository to GitHub.
2. At [render.com](https://render.com), choose **New → Blueprint** and point it at
   the repository. Render reads `render.yaml`.
3. It will prompt for the values marked `sync: false`: `ANTHROPIC_API_KEY` and
   `NCBI_CONTACT_EMAIL`. These are stored as secrets and never enter the image.
4. Deploy. The URL looks like `https://bio-research-copilot.onrender.com`.

**Know this before sending the link to anyone:** free Render services sleep after
a period of inactivity, and the next visitor waits through a cold start of
roughly thirty to sixty seconds before the page appears. For a portfolio link
that is a real cost. A paid instance removes it.

## Hugging Face Spaces

A reasonable alternative, and a recognisable place for machine-learning work. A
Docker Space needs YAML front matter in *its* `README.md`, which conflicts with
this repository's, so push to the Space as a separate remote with a README
containing at least:

```yaml
---
title: Bio Research Copilot
sdk: docker
app_port: 8000
---
```

Add `ANTHROPIC_API_KEY` under the Space's **Settings → Secrets**. Free Spaces
also pause when idle.

## Verifying a deployment

```bash
curl https://YOUR-URL/api/health
```

A healthy service reports `status`, how many recorded examples it holds, whether
a model is available, and what remains of the day's budget:

```json
{"status": "ok", "examples": 12, "model_available": true,
 "budget": {"limit": 200000, "spent": 0, "remaining": 200000}}
```

`model_available: false` means either no key reached the container or the budget
is spent. The demo still works in that state; it answers extractively.

## What is not verified

The container has never been built. Docker is not installed on the machine this
was developed on, so the image is unbuilt and untested, though the start command
it runs has been checked directly and the application starts both with a key and
without one. Expect to fix something on the first build rather than none.
