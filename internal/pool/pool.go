package pool

import (
	"sync"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type ResultPool struct{ pool sync.Pool }

func NewResultPool(preWarm int) *ResultPool {
	rp := &ResultPool{pool: sync.Pool{New: func() any { return &types.MemoryResult{} }}}
	for i := 0; i < preWarm; i++ {
		rp.pool.Put(&types.MemoryResult{})
	}
	return rp
}
func (rp *ResultPool) Get() *types.MemoryResult  { return rp.pool.Get().(*types.MemoryResult) }
func (rp *ResultPool) Put(r *types.MemoryResult) { r.Reset(); rp.pool.Put(r) }

type ResultSlicePool struct{ pool sync.Pool }

func NewResultSlicePool() *ResultSlicePool {
	return &ResultSlicePool{pool: sync.Pool{New: func() any { r := make([]*types.MemoryResult, 0, 64); return &r }}}
}
func (sp *ResultSlicePool) Get() *[]*types.MemoryResult { return sp.pool.Get().(*[]*types.MemoryResult) }
func (sp *ResultSlicePool) Put(s *[]*types.MemoryResult) { *s = (*s)[:0]; sp.pool.Put(s) }
