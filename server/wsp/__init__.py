"""Weight-space programming: read a network's weights, decide from them how to fine-tune it, and edit
weights directly (after Takehi, Teoh, Zhang and Khattab, "Weight-Space Programming", MIT CSAIL, 2026).

finetune.py   the fine-tune planner (no language model needed): which matrices, which rank, checked
              against every-matrix, last-blocks and random choices at the same budget
workspace.py  the fixed tool set an agent reads and edits weights through, with exact undo
agent.py      Claude driving those tools (Messages API tool use)
tasks.py      read quizzes with known answers, and fix tasks judged on held-out examples with a control
api.py        HTTP routes
"""
