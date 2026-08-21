/**
 * WASM Trace Buffer Test
 *
 * Verifies that the WASM trace API works:
 * 1. Enable trace via wasm_cluster_set_trace
 * 2. Process frames
 * 3. Verify trace events are emitted
 * 4. Verify trace count and buffer reads
 */

const { readFileSync } = require('fs');
const { join } = require('path');

async function main() {
  const wasmPath = join(__dirname, '..', 'site',
    'simulator', 'wasm', 'gric_cluster.js');
  const factory = require(wasmPath);
  const Module = await factory();

  const init = Module.cwrap('wasm_cluster_init',
    'number', [
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
    ]);

  const processFrame = Module.cwrap(
    'wasm_cluster_process_frame', 'number',
    ['number', 'number', 'number']);

  const setTrace = Module.cwrap(
    'wasm_cluster_set_trace', null,
    ['number', 'number', 'number']);

  const getTraceCount = Module.cwrap(
    'wasm_cluster_get_trace_count', 'number',
    ['number']);

  const getTraceEvents = Module.cwrap(
    'wasm_cluster_get_trace_events', 'number',
    ['number']);

  const getTraceEventSize = Module.cwrap(
    'wasm_cluster_get_trace_event_size', 'number',
    []);

  const getTraceHead = Module.cwrap(
    'wasm_cluster_get_trace_head', 'number',
    ['number']);

  const getTraceFrameStart = Module.cwrap(
    'wasm_cluster_get_trace_frame_start', 'number',
    ['number']);

  const clearTrace = Module.cwrap(
    'wasm_cluster_clear_trace', null,
    ['number']);

  const freeSession = Module.cwrap(
    'wasm_cluster_free', null, ['number']);

  /* Initialize session */
  const handle = init(
    0.1, 256, 10000, 2,
    0, 0, 0, 0, 2, 0, 0.0,
    0, 0, 0, 0.75, 1.5, 0, 1.0, 0, 0.1
  );

  if (!handle) {
    console.error('ERROR: init returned NULL');
    process.exit(1);
  }

  /* Enable trace */
  setTrace(handle, 1, 2048);
  const eventSize = getTraceEventSize();
  console.log(`TraceEvent size: ${eventSize} bytes`);

  /* Before any frames: count should be 0 */
  let count = getTraceCount(handle);
  if (count !== 0) {
    console.error(
      `FAIL: expected 0 events before frames, got ${count}`);
    process.exit(1);
  }

  /* Process first frame — should create initial cluster */
  const coordsPtr = Module._malloc(2 * 8);
  Module.setValue(coordsPtr, 0.1, 'double');
  Module.setValue(coordsPtr + 8, 0.2, 'double');
  processFrame(handle, coordsPtr, 2);

  count = getTraceCount(handle);
  console.log(`After frame 0: ${count} trace event(s)`);
  if (count < 1) {
    console.error('FAIL: expected at least 1 event');
    process.exit(1);
  }

  /* Read the first event type */
  const eventsPtr = getTraceEvents(handle);
  const firstType = Module.HEAP32[eventsPtr >> 2] & 0xFFFF;
  console.log(`First event type: ${firstType} ` +
    `(expect 1 = TRACE_INITIAL_CLUSTER)`);

  if (firstType !== 1) {
    console.error(
      `FAIL: expected type 1, got ${firstType}`);
    process.exit(1);
  }

  /* Process more frames */
  for (let i = 1; i < 10; i++) {
    const angle = i * 0.6;
    Module.setValue(coordsPtr,
      0.1 + 0.05 * Math.cos(angle), 'double');
    Module.setValue(coordsPtr + 8,
      0.2 + 0.05 * Math.sin(angle), 'double');
    processFrame(handle, coordsPtr, 2);
  }

  count = getTraceCount(handle);
  const head = getTraceHead(handle);
  const frameStart = getTraceFrameStart(handle);
  console.log(
    `After 10 frames: ${count} events, head=${head}, ` +
    `frameStart=${frameStart}`);

  if (count < 10) {
    console.error(
      `FAIL: expected >= 10 events after 10 frames, ` +
      `got ${count}`);
    process.exit(1);
  }

  /* frameStart should be less than head (or wrapped) */
  const currentFrameEvents =
    (head - frameStart + count) % count;
  console.log(
    `Current frame has ${currentFrameEvents} event(s)`);

  /* Clear and verify */
  clearTrace(handle);
  count = getTraceCount(handle);
  if (count !== 0) {
    console.error(
      `FAIL: count should be 0 after clear, got ${count}`);
    process.exit(1);
  }

  /* Disable trace and process a frame - should work */
  setTrace(handle, 0, 0);
  processFrame(handle, coordsPtr, 2);

  Module._free(coordsPtr);
  freeSession(handle);

  console.log('\n✅ WASM trace test PASSED');
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
