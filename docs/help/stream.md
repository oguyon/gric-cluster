# stream

## ROLE
Input Source Selection

## FUNCTION
Specifies that the input is a shared memory stream via ImageStreamIO.

## IMPLEMENTATION
Instead of opening a file, the program attaches to an existing System V
shared memory segment and semaphore set managed by the ImageStreamIO
library. It treats the stream as a circular buffer of frames.

## USE
gric-cluster -stream <stream_name>

## SEE ALSO
- `-cnt2sync`: Enable cnt2 synchronization (increment cnt2 after read)
