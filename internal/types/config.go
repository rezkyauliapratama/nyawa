package types

type Config struct {
	Store  StoreConfig  `yaml:"store"`
	Search SearchConfig `yaml:"search"`
	Pool   PoolConfig   `yaml:"pool"`
}

type StoreConfig struct {
	DBPath      string `yaml:"db_path"`
	MaxMemories int    `yaml:"max_memories"`
}

type SearchConfig struct {
	VectorTopK       int     `yaml:"vector_top_k"`
	FTS5TopK         int     `yaml:"fts5_top_k"`
	RRFK             int     `yaml:"rrf_k"`
	RRFDefaultK      int     `yaml:"rrf_default_k"`
	RecencyWeight    float64 `yaml:"recency_weight"`
	ImportanceWeight float64 `yaml:"importance_weight"`
}

type PoolConfig struct {
	ResultPoolSize int `yaml:"result_pool_size"`
}

func DefaultConfig() Config {
	return Config{
		Store: StoreConfig{DBPath: "nyawa.db", MaxMemories: 1000000},
		Search: SearchConfig{
			VectorTopK: 50, FTS5TopK: 50, RRFK: 60, RRFDefaultK: 60,
			RecencyWeight: 0.05, ImportanceWeight: 0.10,
		},
		Pool: PoolConfig{ResultPoolSize: 64},
	}
}
