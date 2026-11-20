package main

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type batchTestEmbedder struct {
	batches    [][]string
	embedCalls int
	batchFn    func([]string) ([][]float32, error)
}

func (e *batchTestEmbedder) Embed(string) ([]float32, error) {
	e.embedCalls++
	return []float32{1, 2}, nil
}
func (*batchTestEmbedder) Name() string    { return "test" }
func (*batchTestEmbedder) Dims() int       { return 2 }
func (*batchTestEmbedder) Available() bool { return true }
func (e *batchTestEmbedder) EmbedPassages(texts []string) ([][]float32, error) {
	e.batches = append(e.batches, append([]string(nil), texts...))
	if e.batchFn != nil {
		return e.batchFn(texts)
	}
	vectors := make([][]float32, len(texts))
	for i := range vectors {
		vectors[i] = []float32{1, 2}
	}
	return vectors, nil
}

type singleTestEmbedder struct{ calls int }

func (e *singleTestEmbedder) Embed(string) ([]float32, error) { e.calls++; return []float32{1, 2}, nil }
func (*singleTestEmbedder) Name() string                      { return "single" }
func (*singleTestEmbedder) Dims() int                         { return 2 }
func (*singleTestEmbedder) Available() bool                   { return true }

type testVectorIndex struct {
	vectors   map[string][]float32
	insertErr error
}

func newTestVectorIndex() *testVectorIndex {
	return &testVectorIndex{vectors: make(map[string][]float32)}
}
func (i *testVectorIndex) Contains(id string) bool { _, ok := i.vectors[id]; return ok }
func (i *testVectorIndex) Insert(id string, v []float32) error {
	if i.insertErr != nil {
		return i.insertErr
	}
	i.vectors[id] = append([]float32(nil), v...)
	return nil
}
func memories(n int) []*types.Memory {
	ms := make([]*types.Memory, n)
	for i := range ms {
		ms[i] = &types.Memory{ID: fmt.Sprintf("id-%03d", i), Content: fmt.Sprintf("text-%03d", i)}
	}
	return ms
}

func TestReindexMemoriesBatchUsesBatchRequests(t *testing.T) {
	e, idx := &batchTestEmbedder{}, newTestVectorIndex()
	a, r, f, q, err := reindexMemoriesBatch(e, idx, memories(100), 32, 1000000)
	if err != nil || a != 0 || r != 100 || f != 0 || q != 4 {
		t.Fatalf("got (%d,%d,%d,%d), err=%v; want (0,100,0,4)", a, r, f, q, err)
	}
	if len(idx.vectors) != 100 || len(e.batches) != 4 || e.embedCalls != 0 {
		t.Fatalf("inserted=%d batches=%d single=%d", len(idx.vectors), len(e.batches), e.embedCalls)
	}
}
func TestReindexMemoriesBatchSkipsAlreadyIndexed(t *testing.T) {
	ms, idx := memories(4), newTestVectorIndex()
	idx.vectors[ms[1].ID] = []float32{9}
	e := &batchTestEmbedder{}
	a, r, f, q, err := reindexMemoriesBatch(e, idx, ms, 2, 1000000)
	if err != nil || a != 1 || r != 3 || f != 0 || q != 2 {
		t.Fatalf("got (%d,%d,%d,%d), err=%v", a, r, f, q, err)
	}
	for _, b := range e.batches {
		for _, text := range b {
			if text == ms[1].Content {
				t.Fatal("already indexed memory was embedded")
			}
		}
	}
}
func TestReindexMemoriesBatchCountMismatchFailsWholeBatch(t *testing.T) {
	e := &batchTestEmbedder{batchFn: func(texts []string) ([][]float32, error) {
		if texts[0] == "text-000" {
			return [][]float32{{1, 2}}, nil
		}
		vs := make([][]float32, len(texts))
		for i := range vs {
			vs[i] = []float32{1, 2}
		}
		return vs, nil
	}}
	idx := newTestVectorIndex()
	a, r, f, q, err := reindexMemoriesBatch(e, idx, memories(5), 2, 1000000)
	if err != nil || a != 0 || r != 3 || f != 2 || q != 3 {
		t.Fatalf("got (%d,%d,%d,%d), err=%v", a, r, f, q, err)
	}
	if idx.Contains("id-000") || idx.Contains("id-001") || len(idx.vectors) != 3 {
		t.Fatalf("bad count-mismatched batch insert: %#v", idx.vectors)
	}
}
func TestReindexMemoriesBatchDimensionMismatchIsFatal(t *testing.T) {
	want := fmt.Errorf("provider: %w", embedder.ErrEmbeddingDimensionMismatch)
	e := &batchTestEmbedder{batchFn: func([]string) ([][]float32, error) { return nil, want }}
	a, r, f, q, err := reindexMemoriesBatch(e, newTestVectorIndex(), memories(3), 2, 1000000)
	if !errors.Is(err, embedder.ErrEmbeddingDimensionMismatch) || a != 0 || r != 0 || f != 0 || q != 1 {
		t.Fatalf("got (%d,%d,%d,%d), err=%v", a, r, f, q, err)
	}
}
func TestReindexMemoriesBatchFallsBackToSingleEmbed(t *testing.T) {
	e, idx := &singleTestEmbedder{}, newTestVectorIndex()
	a, r, f, q, err := reindexMemoriesBatch(e, idx, memories(7), 3, 1000000)
	if err != nil || a != 0 || r != 7 || f != 0 || q != 7 || e.calls != 7 || len(idx.vectors) != 7 {
		t.Fatalf("got (%d,%d,%d,%d), calls=%d inserted=%d err=%v", a, r, f, q, e.calls, len(idx.vectors), err)
	}
}
func TestReindexMemoriesBatchInsertErrorIsFatal(t *testing.T) {
	want := errors.New("insert failed")
	idx := newTestVectorIndex()
	idx.insertErr = want
	_, _, _, _, err := reindexMemoriesBatch(&batchTestEmbedder{}, idx, memories(1), 32, 1000000)
	if !errors.Is(err, want) {
		t.Fatalf("error=%v, want insert error", err)
	}
}
func TestParseReindexArgsDefaultAndBatchSizeValidation(t *testing.T) {
	p, n, chars, err := parseReindexArgs([]string{"memory.db"})
	if err != nil || p != "memory.db" || n != 32 || chars != 120000 {
		t.Fatalf("default=(%q,%d,%d,%v)", p, n, chars, err)
	}
	for _, n := range []string{"1", "64"} {
		_, got, _, err := parseReindexArgs([]string{"--batch-size", n, "memory.db"})
		if err != nil || got < 1 || got > 64 {
			t.Errorf("valid size %s: got %d, err %v", n, got, err)
		}
	}
	for _, n := range []string{"0", "65"} {
		if _, _, _, err := parseReindexArgs([]string{"--batch-size", n, "memory.db"}); err == nil {
			t.Errorf("invalid size %s accepted", n)
		}
	}
	for _, c := range []string{"999", "2000001"} {
		if _, _, _, err := parseReindexArgs([]string{"--max-batch-chars", c, "memory.db"}); err == nil {
			t.Errorf("invalid char budget %s accepted", c)
		}
	}
}

func longMemories(n, size int) []*types.Memory {
	ms := make([]*types.Memory, n)
	for i := range ms {
		ms[i] = &types.Memory{
			ID:      fmt.Sprintf("long-%03d", i),
			Content: fmt.Sprintf("%03d%s", i, strings.Repeat("x", size)),
		}
	}
	return ms
}

// A provider request must stay inside the client timeout even when every memory
// is long, so the character budget has to split batches before the item cap does.
func TestReindexMemoriesBatchSplitsOnCharacterBudget(t *testing.T) {
	e, idx := &batchTestEmbedder{}, newTestVectorIndex()
	a, r, f, q, err := reindexMemoriesBatch(e, idx, longMemories(10, 4000), 32, 10000)
	if err != nil || a != 0 || r != 10 || f != 0 {
		t.Fatalf("got (%d,%d,%d,%d), err=%v; want (0,10,0,*)", a, r, f, q, err)
	}
	// Each memory is ~4000 chars and the budget is 10000, so at most two fit per
	// request: ten memories need at least five requests, never one.
	if q < 5 {
		t.Fatalf("requests=%d, want >= 5 with a 10000-char budget", q)
	}
	for _, b := range e.batches {
		total := 0
		for _, text := range b {
			total += len(text)
		}
		if len(b) > 1 && total > 10000 {
			t.Fatalf("batch of %d items carried %d chars, over the 10000 budget", len(b), total)
		}
	}
}

// One oversized memory must still be embedded on its own instead of being
// skipped, otherwise the index keeps a silent hole.
func TestReindexMemoriesBatchSendsOversizedMemoryAlone(t *testing.T) {
	e, idx := &batchTestEmbedder{}, newTestVectorIndex()
	a, r, f, q, err := reindexMemoriesBatch(e, idx, longMemories(3, 20000), 32, 5000)
	if err != nil || a != 0 || r != 3 || f != 0 || q != 3 {
		t.Fatalf("got (%d,%d,%d,%d), err=%v; want (0,3,0,3)", a, r, f, q, err)
	}
	for _, b := range e.batches {
		if len(b) != 1 {
			t.Fatalf("oversized memories were grouped: %d items", len(b))
		}
	}
}
