# cnt2sync

## ROLE
Stream Synchronization

## FUNCTION
Enables synchronization using the 'cnt2' counter in ImageStreamIO.

## IMPLEMENTATION
Standard streaming reads whenever a new frame is available (cnt0 increments).
With -cnt2sync, the program waits for the writer to increment 'cnt0', processes
the frame, and then increments 'cnt2'. This allows the writer to wait for the
reader (handshake), ensuring no frames are dropped in a tightly coupled loop.

## USE
gric-cluster -stream my_stream -cnt2sync

## REQUIRES
-stream
  Only meaningful in streaming mode.

## SEE ALSO
- `-stream`: Input is an ImageStreamIO stream
