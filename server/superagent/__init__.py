"""The superagent: the overlay at the bottom right, grown from a guide into an agent
that can also do the work, and that supervises every other agent in Isomorph.

  skills      what it can do: each one is an existing Isomorph endpoint (train, trace,
              profile, run the laws, plan a fine-tune, read or fix with Claude, tune,
              compare, mine, run Autopilot...), called over the local API exactly as the
              page calls it, so the same checks apply. Nothing else: no shell, no code,
              no deleting.
  missions    a goal turned into a plan of skill calls. Ready-made playbooks need no
              key; with a key, Claude plans from your words, revises the plan when a
              step fails, and writes the report at the end. You approve the plan (or
              each step) before anything runs.
  board       every agent and worker at a glance: the job queue, Autopilot, Claude's
              Weight-space runs, Tune sessions, its own missions; with alerts (a failed
              job, an idle session holding memory, a finished campaign) and the
              controls to act on them.
"""
