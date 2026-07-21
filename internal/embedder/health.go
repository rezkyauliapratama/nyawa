package embedder

import (
	"log"
	"time"
)

type HealthCheckRunner struct {
	chain      *PriorityChain
	interval   time.Duration
	lastStatus string
	stopCh     chan struct{}
}

func NewHealthCheckRunner(chain *PriorityChain, interval time.Duration) *HealthCheckRunner {
	if interval <= 0 {
		interval = 60 * time.Second
	}
	return &HealthCheckRunner{chain: chain, interval: interval, lastStatus: "unknown", stopCh: make(chan struct{})}
}

func (h *HealthCheckRunner) Start() { go h.loop() }
func (h *HealthCheckRunner) Stop()  { close(h.stopCh) }

func (h *HealthCheckRunner) loop() {
	h.check()
	ticker := time.NewTicker(h.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			h.check()
		case <-h.stopCh:
			return
		}
	}
}

func (h *HealthCheckRunner) check() {
	name, ok := h.chain.HealthCheck()
	status := "available"
	if !ok {
		status = "unavailable"
	}
	if status != h.lastStatus {
		if ok {
			log.Printf("Embedder health: %s → available (%s)", h.lastStatus, name)
		} else {
			log.Printf("Embedder health: %s → unavailable", h.lastStatus)
		}
		h.lastStatus = status
	}
}

func (h *HealthCheckRunner) Status() string { return h.lastStatus }
