# n-cop Web UI (2-player)

A lightweight, no-build web UI to play the **n-cop Shannon Switching game** on a complete graph.

## Run

Just open the file in a browser:

- `webui/index.html`

No server required.

## Controls

- **Click an edge** to claim it for the current player.
  - **Cop** blocks edges (tracked as “blue”).
  - **Robber** claims edges (red) exactly **1** edge per turn.
- **End Cop turn** ends the Cop turn early (since Cop can play *up to* n edges).
- **Undo** undoes the last move.
- **Auto cop** (top-right): when enabled, the computer will play the cop move using a time-bounded adversarial search.
- **Auto robber** (top-right): when enabled, the computer will play the robber move using a time-bounded adversarial search seeded by the majority/minority component heuristic (fallback: paper strategy).

AI note: use the **Think** selector (Light/Deep) in the top bar to control search depth/time. Deep uses up to ~10 plies and up to 5000ms per move and can temporarily freeze the tab while thinking.
- **Move list**: the left panel includes a chess-like move list (rounds), e.g. `C: (u,v);(x,y)  R: (a,b)`.
- **Position import/export**: use the **Position** box to export the current game (human-readable, using the same move notation) and to import a pasted position.
  - **Copy Link** generates a shareable URL with a `#k=...&n=...&m=...` hash that auto-loads when opened.
- **Animation extract**: click **Extract** to export the **move timeline** from start → current position using the **current cop view**. Choose format: **GIF**, **MP4**, **WebM**, or **animated WebP**. It also adds a short pause on the final position. This lazily downloads ffmpeg.wasm on first use (large download) and requires running over `https://` (works on GitHub Pages). (Shift-click **Extract** to export the view-modes demo animation.)
- **Zoom/Pan**: mouse wheel zooms (around the cursor). Hold **Shift** and drag to pan. Use the `+`, `−`, and `Reset` buttons in the top bar.
- **Cop edges view**:
  - **Show as blue edges**: blocked edges are drawn in blue.
  - **Deletions layout (hide blue)**: blocked edges are hidden so you see the remaining graph. Node positions are recomputed from scratch using a standard force-directed layout: **node repulsion** + **springs on remaining (non-deleted) edges**. Each recompute starts from the same **ring**. Spring rest-lengths are taken from that ring geometry, so the complete graph’s equilibrium is the ring. Then it smoothly animates.
  - **Robber-attraction layout (hide blue)**: same as deletions layout, plus **extra attraction** (stronger/shorter springs) along **red** edges so red-connected vertices prefer to sit closer (repulsion still prevents overlap).

## Win conditions

- **Robber wins** if the red edges contain a path from vertex **0** to **1**.
- **Cop wins** if, in the graph consisting of all **non-blue** edges (neutral + red), vertices **0** and **1** are disconnected.

## Tips

- For larger graphs (k ≥ 10), set **Render neutral edges** to **Only claimed/blocked edges** to reduce clutter.
- Hover a vertex to highlight the edges directly incident to it in the current remaining (non-blue) graph.
