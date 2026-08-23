/**
 * @file fs_access.js
 * @brief W3C File System Access API abstraction for client-side local directory
 *        read/write in the GRIC Interactive Simulator web application.
 */

// eslint-disable-next-line no-unused-vars
const WebFs = (function () {
  'use strict';

  let _dirHandle = null;
  let _dirName = '';

  /**
   * Check if the browser supports the File System Access API directory picker.
   * Supported in Chrome, Edge, Chromium, Opera desktop browsers.
   */
  function isSupported() {
    return typeof window !== 'undefined' &&
           typeof window.showDirectoryPicker === 'function';
  }

  /**
   * Prompt user to select a local directory on their hard drive.
   * Requests readwrite permissions.
   *
   * @returns {Promise<Object|null>} Directory information or null if cancelled.
   */
  async function openDirectory() {
    if (!isSupported()) {
      alert(
        'File System Access API is not supported in this browser.\n' +
        'Please use Chrome, Edge, Chromium, or the Desktop application (gric-gui).'
      );
      return null;
    }

    try {
      _dirHandle = await window.showDirectoryPicker({
        mode: 'readwrite',
        startIn: 'documents'
      });

      _dirName = _dirHandle.name;
      return {
        name: _dirName,
        handle: _dirHandle
      };
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('[WebFs] Error selecting directory:', err);
      }
      return null;
    }
  }

  /**
   * Check if a directory handle is currently open and active.
   */
  function isOpen() {
    return _dirHandle !== null;
  }

  /**
   * Get the name of the currently opened directory.
   */
  function getDirectoryName() {
    return _dirName || '';
  }

  /**
   * List files in the opened local directory.
   * Filters for relevant dataset extensions (.txt, .csv, .dat, .fits, .clusterdat).
   *
   * @returns {Promise<Array<Object>>} List of file objects.
   */
  async function listFiles() {
    if (!_dirHandle) return [];

    const files = [];
    try {
      // eslint-disable-next-line no-undef
      for await (const [name, handle] of _dirHandle.entries()) {
        const isDir = handle.kind === 'directory';
        const dot = name.lastIndexOf('.');
        const ext = dot >= 0 ? name.slice(dot + 1).toLowerCase() : '';

        let isRelevant = false;
        if (isDir && (name.includes('clusterdat') || name.includes('cluster.out'))) {
          isRelevant = true;
        } else if (!isDir) {
          if (['txt', 'csv', 'dat', 'fits', 'log'].includes(ext)) {
            isRelevant = true;
          }
        }

        if (isRelevant) {
          let size = 0;
          if (!isDir) {
            try {
              const f = await handle.getFile();
              size = f.size;
            } catch (e) {
              /* ignore */
            }
          }
          files.push({
            name: name,
            isDir: isDir,
            ext: ext,
            size: size,
            handle: handle
          });
        }
      }
    } catch (err) {
      console.error('[WebFs] Error listing directory entries:', err);
    }

    files.sort((a, b) => a.name.localeCompare(b.name));
    return files;
  }

  /**
   * Read file content as text from the opened directory.
   *
   * @param {string} fileName Relative file name.
   * @returns {Promise<string>} File text content.
   */
  async function readFile(fileName) {
    if (!_dirHandle) {
      throw new Error('No local workspace directory opened.');
    }

    const fileHandle = await _dirHandle.getFileHandle(fileName);
    const file = await fileHandle.getFile();
    return await file.text();
  }

  /**
   * Write text content to a file in the opened directory.
   *
   * @param {string} fileName Relative file name.
   * @param {string} content Text data to write.
   */
  async function writeFile(fileName, content) {
    if (!_dirHandle) {
      throw new Error('No local workspace directory opened.');
    }

    const fileHandle = await _dirHandle.getFileHandle(fileName, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(content);
    await writable.close();
  }

  /**
   * Export clustering results into a <datasetName>.clusterdat/ subfolder
   * directly on the client's hard drive.
   *
   * @param {string} datasetName Base dataset name.
   * @param {Object} artifacts Object mapping file names to string content:
   *   { 'centroids.txt': '...', 'dcc.txt': '...', 'frame_membership.txt': '...', 'cluster_run.log': '...' }
   */
  async function exportClusterDat(datasetName, artifacts) {
    if (!_dirHandle) {
      throw new Error('No local workspace directory opened.');
    }

    const safeName = datasetName.replace(/[^a-zA-Z0-9_-]/g, '_');
    const folderName = `${safeName}.clusterdat`;

    const subDirHandle = await _dirHandle.getDirectoryHandle(folderName, { create: true });

    for (const [fname, content] of Object.entries(artifacts)) {
      if (typeof content === 'string') {
        const fh = await subDirHandle.getFileHandle(fname, { create: true });
        const wr = await fh.createWritable();
        await wr.write(content);
        await wr.close();
      }
    }

    return folderName;
  }

  return {
    isSupported,
    openDirectory,
    isOpen,
    getDirectoryName,
    listFiles,
    readFile,
    writeFile,
    exportClusterDat
  };
})();
