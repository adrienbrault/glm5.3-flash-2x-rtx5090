# Build offline from workspace root with --pull=false --network=none -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2
FROM tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3

WORKDIR /app
# Reject a moved local tag before overwriting reviewed base files.
RUN PYTHONDONTWRITEBYTECODE=1 python3 -c \
    'import hashlib; from pathlib import Path; expected = {"/app/common/metrics.py": "e018ac792da5110a39b0e5b229d4b6a1f3315d504bd5053626ca4b207ad336cf", "/app/backends/exllamav3/model.py": "7b00e54de9449e0c44dbcb8420a5805d142e272eb0872340087e97f2429d7b1c"}; assert all(Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in expected.items()), "reviewed base files changed; refusing whole-file COPY"; assert not Path("/app/tests/test_metrics_live.py").exists(), "reviewed base unexpectedly contains live metrics test"'

COPY docker/tabby-livemetrics-r3/app/common/metrics.py /app/common/metrics.py
COPY docker/tabby-livemetrics-r3/app/backends/exllamav3/model.py /app/backends/exllamav3/model.py
COPY docker/tabby-livemetrics-r3/app/tests/test_metrics_live.py /app/tests/test_metrics_live.py

RUN PYTHONDONTWRITEBYTECODE=1 python3 -c \
    'import hashlib; from pathlib import Path; from common import metrics; assert callable(metrics.start_generation); assert hasattr(metrics._LiveGeneration, "_generated_now"); assert metrics._GENERATED_TOKENS_LIVE.type == "gauge"; assert all(name in metrics.render_metrics() for name in ("tabby_generated_tokens_live_total", "tabby_prompt_tokens_live_total", "tabby_requests_generating", "tabby_generated_tokens_live_drift")); expected = {"/app/common/metrics.py": "f908a31346a738fadf30e81d92fa2ceeddde4f4bbb82ccc31a66a0c00ee2f9c6", "/app/backends/exllamav3/model.py": "6f5efb08a1898431b62d909e0a84c5404ad06bb91328ddf8b9769ec7f7d63abd", "/app/tests/test_metrics_live.py": "27f4a504e540e42feb5bac394978e4148249775d0cd9d10fc1988a936f5dcfb1"}; assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in expected.items()), "live metrics files do not match"'
