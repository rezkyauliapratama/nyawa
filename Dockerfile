# Multi-stage Docker build for Nyawa
# Stage 1: Build Go binary
FROM golang:1.23-alpine AS builder

RUN apk add --no-cache build-base gcc musl-dev

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=1 GOOS=linux go build -tags 'sqlite_fts5' -ldflags='-s -w -extldflags=-static' -o /nyawa ./cmd/nyawa/

# Stage 2: Minimal runtime image
FROM alpine:3.19

RUN apk add --no-cache ca-certificates tzdata sqlite

COPY --from=builder /nyawa /usr/local/bin/nyawa

EXPOSE 3300

ENTRYPOINT ["nyawa"]
CMD ["serve", "/data/memory.db"]
