# docker build -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2 out/
FROM tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2
USER root
LABEL org.openai.splitdev2="immutable per-layer CPU split registrations, Python only"
COPY splitdev2.patch base-sha256.txt src-sha256.txt landing.py test_splitdev.py image-checks.sh /opt/splitdev/
RUN set -eu; \
    P=$(python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("exllamav3").origin).parent.parent)'); \
    cd "$P"; \
    sha256sum -c /opt/splitdev/base-sha256.txt; \
    mkdir -p /opt/splitdev/base/exllamav3/model /opt/splitdev/base/exllamav3/modules; \
    cp exllamav3/model/moe_cpu_host.py /opt/splitdev/base/exllamav3/model/; \
    cp exllamav3/modules/block_sparse_mlp_cpu.py /opt/splitdev/base/exllamav3/modules/; \
    test ! -e exllamav3/model/moe_split_check.py; \
    patch -p1 --fuzz=0 --batch --forward --dry-run < /opt/splitdev/splitdev2.patch; \
    patch -p1 --fuzz=0 --batch --forward < /opt/splitdev/splitdev2.patch; \
    sha256sum -c /opt/splitdev/src-sha256.txt; \
    find "$P/exllamav3" -type f -name '*.pyc' -delete; \
    bash /opt/splitdev/image-checks.sh
# No native source changes: reuse the base extension, no native rebuild.
# A native-changing variant would require TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4.
