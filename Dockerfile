# Build artifact only. Do not run this image against a live Bot or database without
# completing docs/deployment.md and the existing operator preflight.
FROM node:20-bookworm-slim AS web-build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY --from=web-build /web/dist ./web/dist/
# Deliberately no server CMD: the current guarded runtime requires a separately
# attested PostgreSQL and explicit live Telegram startup acknowledgement.
CMD ["python", "-c", "print('Image built; review docs/deployment.md before starting the runtime')"]