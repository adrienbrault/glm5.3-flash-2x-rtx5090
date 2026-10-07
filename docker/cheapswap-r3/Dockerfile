FROM tabbyapi:r828-prompt-lookup-r3
ENV TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4
COPY . /opt/cheapswap/
RUN python /opt/cheapswap/apply_image.py
