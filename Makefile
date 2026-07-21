.PHONY: all build test clean lint release install

APP     := nyawa
GO      ?= go
TAGS    := sqlite_fts5
LDFLAGS := -s -w

all: build test

build:
	$(GO) build -tags "$(TAGS)" -ldflags="$(LDFLAGS)" -o $(APP) ./cmd/$(APP)/

build-race:
	$(GO) build -tags "$(TAGS)" -race -o $(APP) ./cmd/$(APP)/

test:
	$(GO) test -tags "$(TAGS)" -count=1 -race ./...

test-short:
	$(GO) test -tags "$(TAGS)" -count=1 ./...

test-e2e:
	@echo "Running E2E tests..."
	@bash test_nyawa.sh

lint:
	gofmt -s -w .
	$(GO) vet -tags "$(TAGS)" ./...

clean:
	rm -f $(APP)
	rm -f *.db *.db-wal *.db-shm *.db.hnsw

install: build
	install -m 755 $(APP) /usr/local/bin/$(APP)

release: build
	@gzip -k $(APP)
	@mv $(APP).gz $(APP)-linux-amd64.gz

release-arm64:
	GOARCH=arm64 $(GO) build -tags "$(TAGS)" -ldflags="$(LDFLAGS)" -o $(APP)-linux-arm64 ./cmd/$(APP)/
	@gzip -k $(APP)-linux-arm64

commit: lint test build
	@echo "All checks passed. Ready to commit."

help:
	@echo "Targets:"
	@echo "  build        Build binary (14MB)"
	@echo "  test         Run all tests with race detection"
	@echo "  test-short   Run tests without race"
	@echo "  test-e2e     Run E2E test suite"
	@echo "  lint         Format + vet"
	@echo "  clean        Remove build artifacts"
	@echo "  install      Install to /usr/local/bin"
	@echo "  release      Build compressed release binary"
	@echo "  commit       Run lint + test + build before committing"