package rag

import "testing"

func TestLocalCrossEncoderIsOptIn(t *testing.T) {
	for _, test := range []struct {
		value string
		want  bool
	}{
		{"", false},
		{"0", false},
		{"false", false},
		{"1", true},
		{"TRUE", true},
		{" yes ", true},
	} {
		t.Run(test.value, func(t *testing.T) {
			t.Setenv("NYAWA_ENABLE_CROSS_ENCODER", test.value)
			if got := LocalCrossEncoderEnabled(); got != test.want {
				t.Fatalf("LocalCrossEncoderEnabled() = %v, want %v", got, test.want)
			}
		})
	}
}
