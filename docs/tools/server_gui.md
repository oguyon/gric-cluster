# gric-server & gric-gui: Desktop Simulator & Native Micro-Server

`gric-server` is a native C micro-server providing a REST API and static HTTP file serving for the
interactive web simulator. `gric-gui` is a desktop application wrapper.

---

## 1. `gric-server` (Native C HTTP Micro-Server)

`gric-server` runs with zero dependencies (no Python runtime or Node.js required). It serves the
built-in HTML5/WASM simulator UI and exposes REST endpoints for workspace file I/O and running
native clustering CLI tools.

```bash
gric-server [options]
```

### Options
* `-p, --port <port>`: HTTP listen port (default: `8080` or `$GRIC_GUI_PORT`)
* `-d, --dir <path>`: Working workspace directory (default: current working directory)
* `-w, --docs <path>`: HTML documentation directory path (default: auto-detected)
* `-W, --watch-pid <pid>`: Monitor parent process PID and exit when it terminates
* `-t, --idle-timeout <sec>`: Inactivity timeout in seconds (0 = disabled)
* `--auto-shutdown`: Automatically terminate when all browser client tabs disconnect
* `-v, --verbose`: Enable detailed HTTP request logging
* `-h, --help`: Show help screen and exit

---

## 2. `gric-gui` (Desktop Application Launcher)

`gric-gui` manages the lifecycle of `gric-server` and launches an isolated desktop browser window
(Google Chrome, Chromium, or Firefox in app mode) pointing to the interactive simulator.

```bash
tools/gric-gui [options]
```

### Options
* `-p, --port <port>`: HTTP port for local simulator (default: `8080`)
* `-d, --dir <path>`: Workspace directory (default: `$PWD`)
* `-b, --browser <name>`: Preferred browser (`google-chrome`, `chromium`, `firefox`, `default`)
* `-w, --window-size <W,H>`: Initial application window resolution (default: `1600,1000`)
* `-s, --server-only`: Start background HTTP server without opening a browser
* `-k, --kill-server`: Terminate running background HTTP server on the port
* `--status`: Query active background server PID and browser runtime status
