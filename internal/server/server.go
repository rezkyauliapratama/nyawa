func (s *Server) Start() error {
	host := s.config.Host
	if host == "0.0.0.0" { host = "" }
	addr := fmt.Sprintf("%s:%d", host, s.config.Port)
	s.srv = &http.Server{Addr: addr, Handler: s.withMiddleware(s.mux), ReadTimeout: s.config.ReadTimeout, WriteTimeout: s.config.ReadTimeout}
	log.Printf("Nyawa API server on %s", addr)
	go func() {
		time.Sleep(500 * time.Millisecond)
		if _, err := s.pipeline.Search(types.StoreQuery{QueryText: "warmup", Limit: 1}); err != nil { log.Printf("warmup: %v", err) }
	}()
	return s.srv.ListenAndServe()
}