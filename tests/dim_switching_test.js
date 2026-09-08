/**
 * @file dim_switching_test.js
 * @brief Test suite for 2D -> 3D and 2D -> High-D pattern switching and axis clamping.
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

console.log('--- Running Dimension Switching & Axis Clamping Test Suite ---');

// Setup minimal DOM mock environment for state.js
const mockElements = {};
function getOrCreateMockElement(id) {
  if (!mockElements[id]) {
    mockElements[id] = {
      id,
      style: {},
      children: [],
      options: [],
      value: '',
      innerHTML: '',
      getContext: () => ({}),
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
      appendChild(child) {
        this.children.push(child);
      },
      classList: {
        add: () => {},
        remove: () => {},
        toggle: () => {}
      },
      addEventListener: () => {}
    };
  }
  return mockElements[id];
}

const sandbox = {
  console,
  Math,
  parseInt,
  parseFloat,
  isNaN,
  Number,
  String,
  Array,
  ArrayBuffer,
  Float64Array,
  JSON,
  document: {
    getElementById: (id) => getOrCreateMockElement(id),
    querySelectorAll: () => [],
    createElement: (tag) => ({ tag, value: '', textContent: '' })
  },
  window: {}
};
sandbox.window = sandbox;

const stateCode = fs.readFileSync(
  path.join(__dirname, '../docs/simulator/js/state.js'),
  'utf8'
);

vm.createContext(sandbox);
vm.runInContext(stateCode, sandbox);

// -----------------------------------------------------------------------------
// Test 1: In 2D mode, plotDimZ is NOT clamped to 1
// -----------------------------------------------------------------------------
{
  sandbox.currentDim = 2;
  sandbox.plotDimX = 0;
  sandbox.plotDimY = 1;
  sandbox.plotDimZ = 2;

  sandbox.clampPlottingDimensions();

  assert.strictEqual(sandbox.plotDimX, 0, '2D: plotDimX must be 0');
  assert.strictEqual(sandbox.plotDimY, 1, '2D: plotDimY must be 1');
  assert.strictEqual(sandbox.plotDimZ, 2, '2D: plotDimZ must remain >= 2, not clamped to 1');
  console.log('✅ Test 1 Passed: 2D mode preserves plotDimZ = 2 without downward clamping');
}

// -----------------------------------------------------------------------------
// Test 2: Transition from 2D to pure 3D (currentDim = 3)
// -----------------------------------------------------------------------------
{
  // Simulate previous state having been 2D
  sandbox.currentDim = 2;
  sandbox.clampPlottingDimensions();

  // Now switch to 3D
  sandbox.currentDim = 3;
  sandbox.clampPlottingDimensions();

  assert.strictEqual(sandbox.plotDimX, 0, '3D: plotDimX must be 0');
  assert.strictEqual(sandbox.plotDimY, 1, '3D: plotDimY must be 1');
  assert.strictEqual(sandbox.plotDimZ, 2, '3D: plotDimZ must be 2');
  console.log('✅ Test 2 Passed: Transition to 3D strictly sets distinct axes (0, 1, 2)');
}

// -----------------------------------------------------------------------------
// Test 3: Transition from 2D to 128D (currentDim = 128)
// -----------------------------------------------------------------------------
{
  sandbox.currentDim = 2;
  sandbox.clampPlottingDimensions();

  // Switch to 128D
  sandbox.currentDim = 128;
  sandbox.clampPlottingDimensions();

  assert.strictEqual(sandbox.plotDimX, 0, '128D: plotDimX must be 0');
  assert.strictEqual(sandbox.plotDimY, 1, '128D: plotDimY must be 1');
  assert.strictEqual(sandbox.plotDimZ, 2, '128D: plotDimZ must be 2');
  console.log('✅ Test 3 Passed: Transition to 128D correctly defaults to (0, 1, 2)');
}

// -----------------------------------------------------------------------------
// Test 4: Axis Collision Resolution in High-D mode
// -----------------------------------------------------------------------------
{
  // If plotDimZ is somehow set equal to plotDimY (e.g. from corrupt legacy state)
  sandbox.currentDim = 128;
  sandbox.plotDimX = 0;
  sandbox.plotDimY = 1;
  sandbox.plotDimZ = 1; // Collision: Y === Z

  sandbox.clampPlottingDimensions();

  assert.notStrictEqual(sandbox.plotDimZ, sandbox.plotDimY,
    'Z must not equal Y after clamping');
  assert.notStrictEqual(sandbox.plotDimZ, sandbox.plotDimX,
    'Z must not equal X after clamping');
  assert.strictEqual(sandbox.plotDimZ, 2,
    'Colliding Z with X=0, Y=1 should resolve to 2');
  console.log('✅ Test 4 Passed: Axis collision (Z === Y === 1) resolves to distinct Z = 2');
}

// -----------------------------------------------------------------------------
// Test 5: updatePlottingDimSelectorsUI updates UI with distinct dimensions
// -----------------------------------------------------------------------------
{
  sandbox.currentDim = 128;
  sandbox.plotDimX = 0;
  sandbox.plotDimY = 1;
  sandbox.plotDimZ = 1; // Colliding

  sandbox.updatePlottingDimSelectorsUI();

  const selX = getOrCreateMockElement('selectPlotDimX');
  const selY = getOrCreateMockElement('selectPlotDimY');
  const selZ = getOrCreateMockElement('selectPlotDimZ');

  assert.strictEqual(selX.value, '0', 'UI selectPlotDimX should be 0');
  assert.strictEqual(selY.value, '1', 'UI selectPlotDimY should be 1');
  assert.strictEqual(selZ.value, '2', 'UI selectPlotDimZ should be 2, not 1');
  console.log('✅ Test 5 Passed: updatePlottingDimSelectorsUI populates distinct values');
}

// -----------------------------------------------------------------------------
// Test 6: Repeated back-and-forth switching (2D -> 3D -> 2D -> 128D)
// -----------------------------------------------------------------------------
{
  // Start 2D
  sandbox.currentDim = 2;
  sandbox.clampPlottingDimensions();
  assert.strictEqual(sandbox.plotDimZ, 2, 'Roundtrip step 1 (2D): plotDimZ is 2');

  // Switch to 3D
  sandbox.currentDim = 3;
  sandbox.clampPlottingDimensions();
  assert.strictEqual(sandbox.plotDimZ, 2, 'Roundtrip step 2 (3D): plotDimZ is 2');

  // Switch back to 2D
  sandbox.currentDim = 2;
  sandbox.clampPlottingDimensions();
  assert.strictEqual(sandbox.plotDimZ, 2, 'Roundtrip step 3 (2D): plotDimZ is 2');

  // Switch to 128D
  sandbox.currentDim = 128;
  sandbox.clampPlottingDimensions();
  assert.strictEqual(sandbox.plotDimX, 0, 'Roundtrip step 4 (128D): plotDimX is 0');
  assert.strictEqual(sandbox.plotDimY, 1, 'Roundtrip step 4 (128D): plotDimY is 1');
  assert.strictEqual(sandbox.plotDimZ, 2, 'Roundtrip step 4 (128D): plotDimZ is 2');
  console.log('✅ Test 6 Passed: Roundtrip 2D -> 3D -> 2D -> 128D maintains clean axes');
}

// -----------------------------------------------------------------------------
// Test 7: getPlotCoords produces distinct non-collapsed 3D coordinates
// -----------------------------------------------------------------------------
{
  sandbox.currentDim = 3;
  sandbox.highDProjMode = 'raw';
  sandbox.clampPlottingDimensions();

  const pt3D = { x: 0.25, y: 0.50, z: 0.75, coords: [0.25, 0.50, 0.75] };
  const coords = sandbox.getPlotCoords(pt3D);

  assert.strictEqual(coords.x, 0.25, 'Coord X should be 0.25');
  assert.strictEqual(coords.y, 0.50, 'Coord Y should be 0.50');
  assert.strictEqual(coords.z, 0.75, 'Coord Z should be 0.75');
  assert.notStrictEqual(coords.y, coords.z, 'Y and Z must have distinct coordinates (not flat)');
  console.log('✅ Test 7 Passed: getPlotCoords maps distinct 3D coordinates without flattening');
}

// -----------------------------------------------------------------------------
// Test 8: Slot state saving and restoring across dimension changes
// -----------------------------------------------------------------------------
{
  // Slot A: 2D
  sandbox.switchDatasetSlot('A');
  sandbox.currentDim = 2;
  sandbox.clampPlottingDimensions();
  sandbox.saveSlotState('A');

  // Slot B: 128D
  sandbox.switchDatasetSlot('B');
  sandbox.currentDim = 128;
  sandbox.clampPlottingDimensions();
  sandbox.saveSlotState('B');

  // Switch back to Slot A
  sandbox.loadSlotState('A');
  assert.strictEqual(sandbox.currentDim, 2, 'Slot A dim must be 2');
  assert.strictEqual(sandbox.plotDimZ, 2, 'Slot A plotDimZ must remain 2');

  // Switch back to Slot B
  sandbox.loadSlotState('B');
  assert.strictEqual(sandbox.currentDim, 128, 'Slot B dim must be 128');
  assert.strictEqual(sandbox.plotDimX, 0, 'Slot B plotDimX must be 0');
  assert.strictEqual(sandbox.plotDimY, 1, 'Slot B plotDimY must be 1');
  assert.strictEqual(sandbox.plotDimZ, 2, 'Slot B plotDimZ must be 2');
  console.log('✅ Test 8 Passed: Slot state saving and restoring preserves distinct dimensions');
}

console.log('--- ALL DIMENSION SWITCHING TESTS PASSED ---');
