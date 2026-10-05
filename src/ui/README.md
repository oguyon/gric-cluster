# User Interface & Terminal Presentation (Level 3)

## Purpose
Terminal rendering, ANSI styling, command-line argument parsing, table generation, word-wrapping,
and help topic presentation (`libgric_ui`).

## Architectural Layer
- **Level**: Level 3 (Presentation)
- **Public Headers**:
  - `ui/cli/cli_opt.h`: Declarative command-line option parser.
  - `ui/cli/cli_colors.h`: ANSI color codes and conditional terminal coloring.
  - `ui/cli/cli_terminal.h`: Terminal geometry detection and width querying.
  - `ui/cli/cli_theme.h`: Visual theme selection and styling profiles.
  - `ui/cli/cli_wrap.h`: Text and paragraph word-wrapping utilities.
  - `ui/cli/cli_help_view.h`: Formatted CLI help and flag usage presentation.
  - `ui/help/cluster_help.h`: In-depth documentation topics and markdown rendering.
- **Permitted Dependencies**:
  - Level 0 (`base/*`), Level 0 (`third_party/cjson/*`).
  - Standard C library and terminal APIs (`termios.h`, `sys/ioctl.h`).
  - May not be linked into Level 2 compute engines (`libgric`).
