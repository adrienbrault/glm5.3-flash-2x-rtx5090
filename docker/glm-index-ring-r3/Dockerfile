FROM tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2
# docker build -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3 out/
USER root
ENV EXL3_DSA_INDEX_RING=0 EXL3_DSA_INDEX_RING_SHADOW=0
ENV TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 TORCH_EXTENSIONS_DIR=/opt/index-ring-extension
COPY ring3.patch SOURCE-SHA256SUMS PATCHED-SHA256SUMS NATIVE-CHANGED.txt install.sh rebuild_extension.py r929_helpers.py r929-glm53-ring-splitdev.sh /opt/index-ring/
COPY tests/ /opt/index-ring/tests/
RUN cd /app && PYTHONPATH=/app bash /opt/index-ring/install.sh
