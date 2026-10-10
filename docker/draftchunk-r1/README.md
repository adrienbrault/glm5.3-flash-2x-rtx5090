# MTP draft load with the prefill chunk size (r1)

One-line TabbyAPI overlay on the served image. TabbyAPI's `load_model_sync` (`backends/exllamav3/model.py`) loads the MTP draft model with `draft_model.load_gen(...)` without `max_chunk_size`, so the draft's DSA index rings keep ExLlamaV3's default bound of 2,048 tokens while the target model's rings follow `chunk_size`. With `INDEX_RING=1` the generator checks every ring against the chunk size, and with `CHUNK` above 2,048 every boot stopped with `Generator chunk/verify exceeds model.load ring bound` (R959, 2026-10-10, results `2026-10-09-r959-glm53-prefill-chunk-cEQZ71`). The overlay adds `max_chunk_size=self.chunk_size` to that one call; at `CHUNK=2048` the call is unchanged in effect.

- `draftchunk.py`: replaces the draft `load_gen` call only when the unpatched call matches exactly once and `self.draft_model.load_gen(` occurs once; otherwise it exits with an error and the build fails. The result is compiled before it is written.
- `Dockerfile`: runs the patch on the base image's `/app/backends/exllamav3/model.py`, then checks for the new argument, compiles the file and asserts that the patched call occurs once. It does not pin the base files' hashes; the exact-match rule in `draftchunk.py` is the check.

Build from the repository root, on the image with the live token counters ([`docker/tabby-livemetrics-r3/`](../tabby-livemetrics-r3/)); the build context is this directory:

```sh
docker build --pull=false --network=none \
  --build-arg BASE=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2 \
  -f docker/draftchunk-r1/Dockerfile \
  -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1 docker/draftchunk-r1
```

Measured on 2026-10-10 (R959d, results `2026-10-10-r959d-glm53-chunk-confirm-HoWhmg`, write-up [`bench/results/r959-glm53-prefill-chunk.md`](../../bench/results/r959-glm53-prefill-chunk.md)): with this image, `CHUNK=4096` and `EXL3_MOE_CPU_SPLIT_BY_DEVICE=96,102`, the boot log reports `max_chunk=4096` for all 13 index rings (target and draft); cold prefill 1,973 and 1,977 tok/s at 32,738 and 32,683 prompt tokens against 1,443 and 1,449 at chunk 2,048 on the same image. R959e (results `2026-10-10-r959e-glm53-chunk-shadow-2ymbQh`) compared the ring with the paged indexer under served traffic at chunk 3,072 and 4,096: 0 mismatched bytes.
