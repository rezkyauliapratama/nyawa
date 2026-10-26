package compactioneval

import (
	"encoding/json"
	"fmt"
	"sort"
)

var recallCutoffs = [...]int{1, 3, 5}

// Score validates and scores one trace.
func Score(trace Trace) (Result, error) {
	return ScoreTraces([]Trace{trace})
}

// ScoreJSON strictly decodes, validates, and scores one immutable trace JSON
// document. It performs no I/O beyond reading the supplied byte slice.
func ScoreJSON(data []byte) (Result, error) {
	trace, err := DecodeTraceJSON(data)
	if err != nil {
		return Result{}, err
	}
	return Score(trace)
}

// MarshalResultJSON serializes a Result in deterministic struct-field and
// slice order. ScoreTraces already sorts all externally visible audit rows.
func MarshalResultJSON(result Result) ([]byte, error) {
	return json.Marshal(result)
}

// ScoreTraces validates a homogeneous collection of traces and aggregates the
// per-run metrics. It rejects duplicate task/run/condition records, mixed
// conditions, and mixed fingerprint policies instead of silently pooling them.
func ScoreTraces(traces []Trace) (Result, error) {
	if err := ValidateTraces(traces); err != nil {
		return Result{}, err
	}

	var problems ValidationErrors
	condition := traces[0].Condition
	fingerprintPolicyID := traces[0].Environment.FingerprintPolicyID
	for i, trace := range traces[1:] {
		if trace.Condition != condition {
			problems.add(fmt.Sprintf("traces[%d].condition", i+1), "must match every trace in an aggregate result")
		}
		if trace.Environment.FingerprintPolicyID != fingerprintPolicyID {
			problems.add(fmt.Sprintf("traces[%d].environment.fingerprint_policy_id", i+1), "must match every trace in an aggregate result")
		}
	}
	if err := problems.asError(); err != nil {
		return Result{}, err
	}

	ordered := append([]Trace(nil), traces...)
	sort.Slice(ordered, func(i, j int) bool {
		return traceLess(ordered[i], ordered[j])
	})

	result := Result{
		SchemaVersion:       ResultSchemaVersion,
		EvaluatorVersion:    EvaluatorVersion,
		Condition:           condition,
		FingerprintPolicyID: fingerprintPolicyID,
		Runs:                len(ordered),
		CheckpointMetrics:   make([]CheckpointMetrics, 0),
		RepeatGroups:        make([]RepeatGroup, 0),
		RetryDiagnostic:     RetryDiagnostic{Groups: make([]RetryDiagnosticGroup, 0)},
		Warnings:            make([]string, 0),
	}

	var repeatedMacro rateMacro
	var toolErrorMacro rateMacro
	nyawaRecallRuns := make([]recallRunMetrics, 0, len(ordered))
	residualRecallRuns := make([]recallRunMetrics, 0, len(ordered))

	for _, trace := range ordered {
		run := scoreRun(trace)
		result.RepeatedToolOutputs += run.repeatedToolOutputs
		result.EligibleToolOutputs += run.eligibleToolOutputs
		result.PostCompactionRepeatedToolOutputs += run.postCompactionRepeatedToolOutputs
		result.ToolErrors += run.toolErrors
		result.ToolAttempts += run.toolAttempts
		result.PostCompactionToolErrors += run.postCompactionToolErrors
		result.NonplanningNonpollingToolErrors += run.nonplanningNonpollingToolErrors
		result.CancelledToolOutputs += run.cancelledToolOutputs
		result.UnknownToolOutcomes += run.unknownToolOutcomes
		result.CheckpointMetrics = append(result.CheckpointMetrics, run.checkpointMetrics...)
		result.RepeatGroups = append(result.RepeatGroups, run.repeatGroups...)
		result.RetryDiagnostic.RepeatedIdenticalCalls += run.repeatedIdenticalCalls
		result.RetryDiagnostic.Groups = append(result.RetryDiagnostic.Groups, run.retryGroups...)
		result.Warnings = append(result.Warnings, run.warnings...)
		repeatedMacro.add(run.repeatedToolOutputs, run.eligibleToolOutputs)
		toolErrorMacro.add(run.toolErrors, run.toolAttempts)
		nyawaRecallRuns = append(nyawaRecallRuns, run.nyawaRecall)
		residualRecallRuns = append(residualRecallRuns, run.residualRecall)
	}

	result.RepeatedToolOutputRateMicro = integerRatio(result.RepeatedToolOutputs, result.EligibleToolOutputs)
	result.RepeatedToolOutputRateMacro = repeatedMacro.value()
	result.RepeatedToolOutputRateMacroContributingRuns = repeatedMacro.count
	result.ToolErrorRateMicro = integerRatio(result.ToolErrors, result.ToolAttempts)
	result.ToolErrorRateMacro = toolErrorMacro.value()
	result.ToolErrorRateMacroContributingRuns = toolErrorMacro.count
	result.RecallAfterCompaction = aggregateRecallMetrics(nyawaRecallRuns)
	result.ResidualPackLookupAfterCompaction = aggregateRecallMetrics(residualRecallRuns)

	sort.Slice(result.CheckpointMetrics, func(i, j int) bool {
		return checkpointMetricsLess(result.CheckpointMetrics[i], result.CheckpointMetrics[j])
	})
	sort.Slice(result.RepeatGroups, func(i, j int) bool {
		return repeatGroupLess(result.RepeatGroups[i], result.RepeatGroups[j])
	})
	sort.Slice(result.RetryDiagnostic.Groups, func(i, j int) bool {
		return retryGroupLess(result.RetryDiagnostic.Groups[i], result.RetryDiagnostic.Groups[j])
	})
	sort.Strings(result.Warnings)

	return result, nil
}

type runMetrics struct {
	repeatedToolOutputs               int
	eligibleToolOutputs               int
	postCompactionRepeatedToolOutputs int
	toolErrors                        int
	toolAttempts                      int
	postCompactionToolErrors          int
	nonplanningNonpollingToolErrors   int
	cancelledToolOutputs              int
	unknownToolOutcomes               int
	repeatedIdenticalCalls            int
	checkpointMetrics                 []CheckpointMetrics
	repeatGroups                      []RepeatGroup
	retryGroups                       []RetryDiagnosticGroup
	nyawaRecall                       recallRunMetrics
	residualRecall                    recallRunMetrics
	warnings                          []string
}

type outputGroupKey struct {
	toolName     string
	outputSHA256 string
}

type retryGroupKey struct {
	toolName     string
	inputSHA256  string
	hasInput     bool
	outputSHA256 string
}

type repeatGroupAccumulator struct {
	count           int
	firstEventIndex int
	repeats         int
	attributions    map[string]*repeatAttributionAccumulator
}

type repeatAttributionAccumulator struct {
	checkpointID *string
	eventCount   int
	repeats      int
}

type retryGroupAccumulator struct {
	count           int
	firstEventIndex int
}

func scoreRun(trace Trace) runMetrics {
	run := runMetrics{
		checkpointMetrics: make([]CheckpointMetrics, len(trace.Compactions)),
		warnings:          make([]string, 0),
	}
	for i, compaction := range trace.Compactions {
		run.checkpointMetrics[i] = CheckpointMetrics{
			SuiteID:            trace.SuiteID,
			TaskID:             trace.TaskID,
			RunID:              trace.RunID,
			CheckpointID:       compaction.CheckpointID,
			BoundaryEventIndex: compaction.EventIndex,
		}
	}

	repeatAccumulators := make(map[outputGroupKey]*repeatGroupAccumulator)
	retryAccumulators := make(map[retryGroupKey]*retryGroupAccumulator)
	currentCheckpoint := -1
	nextCheckpoint := 0

	for _, event := range trace.ToolEvents {
		for nextCheckpoint < len(trace.Compactions) && trace.Compactions[nextCheckpoint].EventIndex <= event.EventIndex {
			currentCheckpoint = nextCheckpoint
			nextCheckpoint++
		}

		var attributedCheckpointID *string
		if currentCheckpoint >= 0 {
			attributedCheckpointID = stringPointer(trace.Compactions[currentCheckpoint].CheckpointID)
		}

		run.toolAttempts++
		if currentCheckpoint >= 0 {
			run.checkpointMetrics[currentCheckpoint].ToolAttempts++
		}

		if isToolError(event.Outcome) {
			run.toolErrors++
			if currentCheckpoint >= 0 {
				run.postCompactionToolErrors++
				run.checkpointMetrics[currentCheckpoint].ToolErrors++
			}
			if !event.IsPlanning && !event.IsPolling {
				run.nonplanningNonpollingToolErrors++
				if currentCheckpoint >= 0 {
					run.checkpointMetrics[currentCheckpoint].NonplanningNonpollingToolErrors++
				}
			}
		}
		switch event.Outcome {
		case OutcomeCancelled:
			run.cancelledToolOutputs++
			if currentCheckpoint >= 0 {
				run.checkpointMetrics[currentCheckpoint].CancelledToolOutputs++
			}
		case OutcomeUnknown:
			run.unknownToolOutcomes++
			if currentCheckpoint >= 0 {
				run.checkpointMetrics[currentCheckpoint].UnknownToolOutcomes++
			}
		}

		if event.IsPlanning || event.IsPolling || event.OutputSHA256 == nil {
			continue
		}

		run.eligibleToolOutputs++
		if currentCheckpoint >= 0 {
			run.checkpointMetrics[currentCheckpoint].EligibleToolOutputs++
		}

		outputKey := outputGroupKey{toolName: event.ToolName, outputSHA256: *event.OutputSHA256}
		group := repeatAccumulators[outputKey]
		if group == nil {
			group = &repeatGroupAccumulator{
				firstEventIndex: event.EventIndex,
				attributions:    make(map[string]*repeatAttributionAccumulator),
			}
			repeatAccumulators[outputKey] = group
		}
		isRepeat := group.count > 0
		group.count++
		group.addAttribution(attributedCheckpointID, isRepeat)
		if isRepeat {
			run.repeatedToolOutputs++
			if currentCheckpoint >= 0 {
				run.postCompactionRepeatedToolOutputs++
				run.checkpointMetrics[currentCheckpoint].RepeatedToolOutputs++
			}
		}

		retryKey := retryGroupKey{
			toolName:     event.ToolName,
			outputSHA256: *event.OutputSHA256,
		}
		if event.ToolInputSHA256 != nil {
			retryKey.hasInput = true
			retryKey.inputSHA256 = *event.ToolInputSHA256
		}
		retryGroup := retryAccumulators[retryKey]
		if retryGroup == nil {
			retryGroup = &retryGroupAccumulator{firstEventIndex: event.EventIndex}
			retryAccumulators[retryKey] = retryGroup
		}
		if retryGroup.count > 0 {
			run.repeatedIdenticalCalls++
		}
		retryGroup.count++
	}

	for _, probe := range trace.RecallProbes {
		switch probe.Surface {
		case RecallSurfaceNyawa:
			run.nyawaRecall.add(probe, probe.TargetMemoryIDs)
		case RecallSurfaceResidualPack:
			run.residualRecall.add(probe, probe.TargetResidualEntryIDs)
		}
	}

	run.repeatGroups = makeRepeatGroups(trace, repeatAccumulators)
	run.retryGroups = makeRetryGroups(trace, retryAccumulators)
	if run.unknownToolOutcomes > 0 {
		run.warnings = append(run.warnings, fmt.Sprintf("%s: %d tool event(s) have unknown outcomes", traceIdentity(trace), run.unknownToolOutcomes))
	}
	return run
}

func (group *repeatGroupAccumulator) addAttribution(checkpointID *string, isRepeat bool) {
	key := checkpointAttributionKey(checkpointID)
	attribution := group.attributions[key]
	if attribution == nil {
		attribution = &repeatAttributionAccumulator{checkpointID: cloneStringPointer(checkpointID)}
		group.attributions[key] = attribution
	}
	attribution.eventCount++
	if isRepeat {
		group.repeats++
		attribution.repeats++
	}
}

func makeRepeatGroups(trace Trace, accumulators map[outputGroupKey]*repeatGroupAccumulator) []RepeatGroup {
	keys := make([]outputGroupKey, 0, len(accumulators))
	for key := range accumulators {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].toolName != keys[j].toolName {
			return keys[i].toolName < keys[j].toolName
		}
		return keys[i].outputSHA256 < keys[j].outputSHA256
	})

	groups := make([]RepeatGroup, 0, len(keys))
	for _, key := range keys {
		accumulator := accumulators[key]
		attributions := make([]RepeatCheckpointAttribution, 0, len(accumulator.attributions))
		for _, attribution := range accumulator.attributions {
			attributions = append(attributions, RepeatCheckpointAttribution{
				CheckpointID:       cloneStringPointer(attribution.checkpointID),
				EventCount:         attribution.eventCount,
				RepeatContribution: attribution.repeats,
			})
		}
		sort.Slice(attributions, func(i, j int) bool {
			return repeatAttributionLess(attributions[i], attributions[j])
		})
		groups = append(groups, RepeatGroup{
			SuiteID:                trace.SuiteID,
			TaskID:                 trace.TaskID,
			RunID:                  trace.RunID,
			ToolName:               key.toolName,
			OutputSHA256:           key.outputSHA256,
			Count:                  accumulator.count,
			FirstEventIndex:        accumulator.firstEventIndex,
			RepeatContribution:     accumulator.repeats,
			CheckpointAttributions: attributions,
		})
	}
	return groups
}

func makeRetryGroups(trace Trace, accumulators map[retryGroupKey]*retryGroupAccumulator) []RetryDiagnosticGroup {
	keys := make([]retryGroupKey, 0, len(accumulators))
	for key := range accumulators {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].toolName != keys[j].toolName {
			return keys[i].toolName < keys[j].toolName
		}
		if keys[i].hasInput != keys[j].hasInput {
			return !keys[i].hasInput
		}
		if keys[i].inputSHA256 != keys[j].inputSHA256 {
			return keys[i].inputSHA256 < keys[j].inputSHA256
		}
		return keys[i].outputSHA256 < keys[j].outputSHA256
	})

	groups := make([]RetryDiagnosticGroup, 0, len(keys))
	for _, key := range keys {
		accumulator := accumulators[key]
		var input *string
		if key.hasInput {
			input = stringPointer(key.inputSHA256)
		}
		groups = append(groups, RetryDiagnosticGroup{
			SuiteID:           trace.SuiteID,
			TaskID:            trace.TaskID,
			RunID:             trace.RunID,
			ToolName:          key.toolName,
			ToolInputSHA256:   input,
			OutputSHA256:      key.outputSHA256,
			Count:             accumulator.count,
			FirstEventIndex:   accumulator.firstEventIndex,
			RetryContribution: maxZero(accumulator.count - 1),
		})
	}
	return groups
}

type recallRunMetrics struct {
	valid             int
	unobservable      int
	superseded        int
	hits              [3]int
	denominators      [3]int
	reciprocalRankSum float64
	mrrDenominator    int
}

func (metrics *recallRunMetrics) add(probe RecallProbe, targets []string) {
	switch effectiveTargetStatus(probe.TargetStatus) {
	case RecallTargetUnobservable:
		metrics.unobservable++
		return
	case RecallTargetSuperseded:
		metrics.superseded++
		return
	}

	metrics.valid++
	firstRank := firstTargetRank(probe.ResultIDs, targets)
	for i, cutoff := range recallCutoffs {
		if probe.Limit < cutoff {
			continue
		}
		metrics.denominators[i]++
		if firstRank > 0 && firstRank <= cutoff {
			metrics.hits[i]++
		}
	}
	if probe.Limit >= 5 {
		metrics.mrrDenominator++
		if firstRank > 0 && firstRank <= 5 {
			metrics.reciprocalRankSum += 1.0 / float64(firstRank)
		}
	}
}

func aggregateRecallMetrics(runMetrics []recallRunMetrics) RecallMetrics {
	var aggregate recallRunMetrics
	var recallMacros [3]rateMacro
	var mrrMacro rateMacro

	for _, run := range runMetrics {
		aggregate.valid += run.valid
		aggregate.unobservable += run.unobservable
		aggregate.superseded += run.superseded
		for i := range recallCutoffs {
			aggregate.hits[i] += run.hits[i]
			aggregate.denominators[i] += run.denominators[i]
			recallMacros[i].add(run.hits[i], run.denominators[i])
		}
		aggregate.reciprocalRankSum += run.reciprocalRankSum
		aggregate.mrrDenominator += run.mrrDenominator
		mrrMacro.addFloat(run.reciprocalRankSum, run.mrrDenominator)
	}

	return RecallMetrics{
		ValidRecallProbes:              aggregate.valid,
		UnobservableRecallProbes:       aggregate.unobservable,
		SupersededTargetProbes:         aggregate.superseded,
		RecallAt1:                      integerRatio(aggregate.hits[0], aggregate.denominators[0]),
		RecallAt1ContributingProbes:    aggregate.denominators[0],
		RecallAt1Macro:                 recallMacros[0].value(),
		RecallAt1MacroContributingRuns: recallMacros[0].count,
		RecallAt3:                      integerRatio(aggregate.hits[1], aggregate.denominators[1]),
		RecallAt3ContributingProbes:    aggregate.denominators[1],
		RecallAt3Macro:                 recallMacros[1].value(),
		RecallAt3MacroContributingRuns: recallMacros[1].count,
		RecallAt5:                      integerRatio(aggregate.hits[2], aggregate.denominators[2]),
		RecallAt5ContributingProbes:    aggregate.denominators[2],
		RecallAt5Macro:                 recallMacros[2].value(),
		RecallAt5MacroContributingRuns: recallMacros[2].count,
		MRRAt5:                         floatRatio(aggregate.reciprocalRankSum, aggregate.mrrDenominator),
		MRRAt5ContributingProbes:       aggregate.mrrDenominator,
		MRRAt5Macro:                    mrrMacro.value(),
		MRRAt5MacroContributingRuns:    mrrMacro.count,
	}
}

type rateMacro struct {
	sum   float64
	count int
}

func (macro *rateMacro) add(numerator, denominator int) {
	if denominator == 0 {
		return
	}
	macro.sum += float64(numerator) / float64(denominator)
	macro.count++
}

func (macro *rateMacro) addFloat(numerator float64, denominator int) {
	if denominator == 0 {
		return
	}
	macro.sum += numerator / float64(denominator)
	macro.count++
}

func (macro rateMacro) value() *float64 {
	if macro.count == 0 {
		return nil
	}
	value := macro.sum / float64(macro.count)
	return &value
}

func integerRatio(numerator, denominator int) *float64 {
	if denominator == 0 {
		return nil
	}
	value := float64(numerator) / float64(denominator)
	return &value
}

func floatRatio(numerator float64, denominator int) *float64 {
	if denominator == 0 {
		return nil
	}
	value := numerator / float64(denominator)
	return &value
}

func firstTargetRank(resultIDs, targets []string) int {
	targetSet := make(map[string]struct{}, len(targets))
	for _, target := range targets {
		targetSet[target] = struct{}{}
	}
	for i, resultID := range resultIDs {
		if _, found := targetSet[resultID]; found {
			return i + 1
		}
	}
	return 0
}

func effectiveTargetStatus(status RecallTargetStatus) RecallTargetStatus {
	if status == "" {
		return RecallTargetActive
	}
	return status
}

func isToolError(outcome Outcome) bool {
	return outcome == OutcomeFailure || outcome == OutcomeToolError || outcome == OutcomeTimeout
}

func checkpointAttributionKey(checkpointID *string) string {
	if checkpointID == nil {
		return "\x00"
	}
	return "\x01" + *checkpointID
}

func cloneStringPointer(value *string) *string {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func stringPointer(value string) *string {
	copy := value
	return &copy
}

func traceIdentity(trace Trace) string {
	return trace.SuiteID + "/" + trace.TaskID + "/" + trace.RunID + "/" + trace.Condition
}

func traceLess(left, right Trace) bool {
	if left.SuiteID != right.SuiteID {
		return left.SuiteID < right.SuiteID
	}
	if left.TaskID != right.TaskID {
		return left.TaskID < right.TaskID
	}
	if left.RunID != right.RunID {
		return left.RunID < right.RunID
	}
	return left.Condition < right.Condition
}

func checkpointMetricsLess(left, right CheckpointMetrics) bool {
	if left.SuiteID != right.SuiteID {
		return left.SuiteID < right.SuiteID
	}
	if left.TaskID != right.TaskID {
		return left.TaskID < right.TaskID
	}
	if left.RunID != right.RunID {
		return left.RunID < right.RunID
	}
	if left.BoundaryEventIndex != right.BoundaryEventIndex {
		return left.BoundaryEventIndex < right.BoundaryEventIndex
	}
	return left.CheckpointID < right.CheckpointID
}

func repeatGroupLess(left, right RepeatGroup) bool {
	if left.SuiteID != right.SuiteID {
		return left.SuiteID < right.SuiteID
	}
	if left.TaskID != right.TaskID {
		return left.TaskID < right.TaskID
	}
	if left.RunID != right.RunID {
		return left.RunID < right.RunID
	}
	if left.ToolName != right.ToolName {
		return left.ToolName < right.ToolName
	}
	return left.OutputSHA256 < right.OutputSHA256
}

func retryGroupLess(left, right RetryDiagnosticGroup) bool {
	if left.SuiteID != right.SuiteID {
		return left.SuiteID < right.SuiteID
	}
	if left.TaskID != right.TaskID {
		return left.TaskID < right.TaskID
	}
	if left.RunID != right.RunID {
		return left.RunID < right.RunID
	}
	if left.ToolName != right.ToolName {
		return left.ToolName < right.ToolName
	}
	if (left.ToolInputSHA256 == nil) != (right.ToolInputSHA256 == nil) {
		return left.ToolInputSHA256 == nil
	}
	if left.ToolInputSHA256 != nil && *left.ToolInputSHA256 != *right.ToolInputSHA256 {
		return *left.ToolInputSHA256 < *right.ToolInputSHA256
	}
	return left.OutputSHA256 < right.OutputSHA256
}

func repeatAttributionLess(left, right RepeatCheckpointAttribution) bool {
	if (left.CheckpointID == nil) != (right.CheckpointID == nil) {
		return left.CheckpointID == nil
	}
	if left.CheckpointID == nil {
		return false
	}
	return *left.CheckpointID < *right.CheckpointID
}

func maxZero(value int) int {
	if value < 0 {
		return 0
	}
	return value
}
