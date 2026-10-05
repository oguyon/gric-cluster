/**
 * @file status_terminal.c
 * @brief Terminal mode control and signal handling for gric-status TUI.
 */

#include "status_internal.h"
#include <signal.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

/**
 * handle_sigint() - Signal handler for SIGINT and SIGTERM.
 * @sig: Received signal number (unused).
 *
 * Sets the sig_quit flag to trigger a clean exit from the TUI main loop.
 */
void handle_sigint(
    int sig)
{
    (void)sig;
    sig_quit = 1;
}

/**
 * tui_exit_raw_mode() - Restores the terminal settings to original state.
 *
 * Checks if raw mode is active and restores the termios settings stored
 * in orig_termios.
 */
void tui_exit_raw_mode(void)
{
    if (raw_active)
    {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        raw_active = 0;
    }
}

/**
 * tui_enter_raw_mode() - Configures terminal for raw, non-canonical input.
 *
 * Saves the original terminal attributes and configures raw mode settings
 * (disabling ECHO, ICANON, signals, flow control) for the TUI interface.
 */
void tui_enter_raw_mode(void)
{
    if (tcgetattr(STDIN_FILENO, &orig_termios) < 0)
    {
        return;
    }

    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON | ISIG | IEXTEN);
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_cflag |= (CS8);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;

    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) >= 0)
    {
        raw_active = 1;
    }
}

/**
 * crash_handler() - Handles fatal signals by restoring terminal state.
 * @sig: The crash signal number to be re-raised after restoration.
 *
 * Restores canonical terminal mode, cursor visibility, clears alternative
 * screen buffer, resets default signal actions, and re-raises the signal.
 */
void crash_handler(
    int sig)
{
    tui_exit_raw_mode();
    printf("\033[?1049l\033[?25h\033[0m\n");
    fflush(stdout);

    struct sigaction sa;
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(sig, &sa, NULL);
    raise(sig);
}

/**
 * setup_signals() - Registers handlers for standard and crash signals.
 *
 * Hooks SIGINT/SIGTERM for clean shutdown, and SIGSEGV/SIGABRT for terminal
 * restoration on crash.
 */
void setup_signals(void)
{
    struct sigaction sa;
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    struct sigaction sa_crash;
    sa_crash.sa_handler = crash_handler;
    sigemptyset(&sa_crash.sa_mask);
    sa_crash.sa_flags = 0;
    sigaction(SIGSEGV, &sa_crash, NULL);
    sigaction(SIGABRT, &sa_crash, NULL);
}
