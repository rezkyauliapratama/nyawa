.PHONY: build build-fts5 clean test run

build:
	go build -o nyawa ./cmd/nyawa/

build-fts5:
	go build -tags "sqlite_fts5" -o nyawa ./cmd/nyawa/

clean:
	rm -f nyawa
	rm -f *.db

test:
	go test -tags "sqlite_fts5" -v ./...

test-bench:
	go test -tags "sqlite_fts5" -bench=. -benchmem ./...

init:
	./nyawa init nyawa.db

run:
	go run -tags "sqlite_fts5" ./cmd/nyawa/
