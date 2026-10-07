# glm-agent-r2 on cheapswap-r3 (R882, 2026-10-07)

`tabbyapi:cheapswap-r3-agent-r2` combines two packages:

- the r861 agent overlay r2, which patches TabbyAPI under `/app`;
- `tabbyapi:cheapswap-r3`, which patches ExLlamaV3 in the venv.

The two touch disjoint trees. The Dockerfile is `../glm-agent-r2/Dockerfile` with only the `FROM` line and a label changed. The overlay's own checksum of the stock `/app` (SHA256SUMS.base) still gates the build.

Build context: the `glm-agent-r2` package directory.

    cp glm-agent-r2-on-cheapswap-r3/Dockerfile glm-agent-r2/Dockerfile.combo
    docker build -f glm-agent-r2/Dockerfile.combo -t tabbyapi:cheapswap-r3-agent-r2 glm-agent-r2

The Dockerfile must sit inside the context: with -f outside it, the legacy builder rewrites .dockerignore and the checksum gate fails (R882).
