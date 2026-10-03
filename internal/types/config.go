package types

type Config struct {
	Store  StoreConfig  `yaml:"store"`
	Search SearchConfig `yaml:"search"`
	Pool   PoolConfig   `yaml:"pool"`
}

type StoreConfig struct {
	DBPath       string `yaml:"db_path"`
	MaxMemories  int    `yaml:"max_memories"`
}

type SearchConfig struct {
	VectorTopK       int     `yaml:"vector_top_k"`
	FTS5TopK         int     `yaml:"fts5_top_k"`
	RRFK             int     `yaml:"rrf_k"`
	RRFDefaultK      int     `yaml:"rrf_default_k"`
	RecencyWeight    float64 `yaml:"recency_weight"`
	ImportanceWeight float64 `yaml:"importance_weight"`
	// Retrieval-v2 weights (see internal/search/weights.go). A zero value means
	// "use the built-in default", so existing SearchConfig literals keep working.
	// The whole v2 path defaults on and is toggled per-process with the
	// NYAWA_RECALL_V2 env var (0/false/off restores pre-v2 ranking).
	TypeWeight   float64 `yaml:"type_weight"`
	AccessWeight float64 `yaml:"access_weight"`
	LengthWeight float64 `yaml:"length_weight"`
	PinnedWeight float64 `yaml:"pinned_weight"`
	GraphWeight  float64 `yaml:"graph_weight"`
}

type PoolConfig struct {
	ResultPoolSize int `yaml:"result_pool_size"`
}

func DefaultConfig() Config {
	return Config{
		Store: StoreConfig{DBPath: "nyawa.db", MaxMemories: 1000000},
		Search: SearchConfig{
			VectorTopK: 50, FTS5TopK: 50, RRFK: 5, RRFDefaultK: 5,
			RecencyWeight: 0.05, ImportanceWeight: 0.10,
			TypeWeight: 0.12, AccessWeight: 0.05, LengthWeight: 0.05,
			PinnedWeight: 0.10, GraphWeight: 0.03,
		},
		Pool: PoolConfig{ResultPoolSize: 64},
	}
}
