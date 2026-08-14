# conf

## ROLE
Configuration Management

## FUNCTION
Reads clustering options from a configuration
file. Options in the file use the same names
as command-line flags (without the leading
dash).

## FORMAT
One option per line:
  dprob 0.02
  maxcl 500
  gprob
  entropy
Lines starting with '#' are comments.
Command-line options override file values.

## USE
-conf my_run.conf

## SEE ALSO
- `-confw`: Write current options to file
