# input

## OVERVIEW
GRIC accepts frames from several input sources.
The format is auto-detected from the file extension
unless overridden by a flag.

## SUPPORTED FORMATS
Text (.txt)
  One frame per line, space-separated coordinates.
  Simplest format; useful for low-dimensional data.

FITS cube (.fits, .fits.fz)
  3D data cube (width x height x N_frames).
  Standard in astronomy. Requires CFITSIO.

MP4 video (.mp4)
  Video frames extracted as pixel arrays.
  Requires FFmpeg (libav*).

ImageStreamIO (-stream)
  Shared memory circular buffer for real-time
  streaming. Use -cnt2sync for handshake mode.

## THE RLIM PARAMETER
The first positional argument sets the distance
threshold for cluster membership:
  0.5        Literal value
  a1.5       Auto-mode: 1.5 x median distance
             (run -scandist first to calibrate)

Use -scandist to measure distance statistics and
pick a good rlim before a full clustering run.

## SEE ALSO
- `-stream`: Input is an ImageStreamIO stream
- `-cnt2sync`: Enable cnt2 synchronization
- `-scandist`: Measure distance stats (pick rlim)
