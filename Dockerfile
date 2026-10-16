# The command-line scanner in a small image. Only numpy is installed; the scanner
# never imports torch and never unpickles anything.
#   docker build -t stowaway .
#   docker run --rm -v "$PWD:/data" stowaway scan /data/model.safetensors
#   docker run --rm stowaway scan hf:org/finetune --base declared
FROM python:3.12-slim
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY stowaway ./stowaway
RUN pip install --no-cache-dir . && rm -rf /src
RUN useradd --create-home scanner
USER scanner
WORKDIR /data
ENTRYPOINT ["stowaway"]
CMD ["--help"]
