import { unzipSync } from "fflate";

function readAscii(bytes, start, end) {
    return new TextDecoder("latin1").decode(bytes.subarray(start, end));
}

function parseNpy(input) {
    // input MUST be a Uint8Array view (fflate may return views with a byteOffset)
    const bytes = input;

    const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
    if (bytes.length < 10 || magic.some((b, i) => bytes[i] !== b)) {
        throw new Error("Invalid NPY file");
    }

    const major = bytes[6];
    const minor = bytes[7];
    let headerLength, headerStart;

    if (major === 1) {
        headerLength = bytes[8] | (bytes[9] << 8);
        headerStart = 10;
    } else if (major === 2 || major === 3) {
        headerLength = bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
        headerStart = 12;
    } else {
        throw new Error(`Unsupported NPY version ${major}.${minor}`);
    }

    const header = readAscii(bytes, headerStart, headerStart + headerLength);
    const descrMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
    const shapeMatch = header.match(/'shape'\s*:\s*\(([^)]*)\)/);
    const fortranMatch = header.match(/'fortran_order'\s*:\s*(True|False)/);

    if (!descrMatch || !shapeMatch) throw new Error("Could not read NPY header");

    const descr = descrMatch[1];
    const shapeText = shapeMatch[1].trim();
    const shape = shapeText.length === 0
        ? []
        : shapeText.split(",").map((v) => v.trim()).filter(Boolean).map(Number);

    if (fortranMatch && fortranMatch[1] === "True") {
        throw new Error("Fortran-order NPY arrays are not supported");
    }

    const dataStart = headerStart + headerLength;
    const dataBytes = bytes.subarray(dataStart);

    const endian = descr[0];
    const kind = descr[1];
    const itemSize = Number(descr.slice(2));
    const littleFile = endian === "<" || endian === "|" || (endian === "=" && isLittleEndian());

    // ---- string dtypes (<U21 unicode, |S10 bytes) -> array of JS strings ----
    if (kind === "U" || kind === "S") {
        const count = shape.reduce((a, b) => a * b, 1);
        const charBytes = kind === "U" ? 4 : 1;
        const stride = itemSize * charBytes;
        const view = new DataView(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength);
        const strings = [];
        for (let i = 0; i < count; i++) {
            let s = "";
            for (let c = 0; c < itemSize; c++) {
                const off = i * stride + c * charBytes;
                const code = kind === "U" ? view.getUint32(off, littleFile) : view.getUint8(off);
                if (code === 0) break; // numpy pads with NULs
                s += String.fromCodePoint(code);
            }
            strings.push(s);
        }
        return { data: strings, shape, dtype: descr, isString: true };
    }

    let TypedArray;
    if (kind === "f" && itemSize === 4) TypedArray = Float32Array;
    else if (kind === "f" && itemSize === 8) TypedArray = Float64Array;
    else if (kind === "i" && itemSize === 1) TypedArray = Int8Array;
    else if (kind === "i" && itemSize === 2) TypedArray = Int16Array;
    else if (kind === "i" && itemSize === 4) TypedArray = Int32Array;
    else if (kind === "i" && itemSize === 8) TypedArray = BigInt64Array;
    else if (kind === "u" && itemSize === 1) TypedArray = Uint8Array;
    else if (kind === "u" && itemSize === 2) TypedArray = Uint16Array;
    else if (kind === "u" && itemSize === 4) TypedArray = Uint32Array;
    else if (kind === "u" && itemSize === 8) TypedArray = BigUint64Array;
    else if (kind === "b" && itemSize === 1) TypedArray = Uint8Array;
    else throw new Error(`Unsupported NPY dtype: ${descr}`);

    if (dataBytes.byteLength % itemSize !== 0) {
        throw new Error(`Invalid NPY data length for dtype ${descr}`);
    }

    // copy into a fresh, aligned buffer
    const raw = dataBytes.buffer.slice(dataBytes.byteOffset, dataBytes.byteOffset + dataBytes.byteLength);
    let data = new TypedArray(raw);

    const needsByteSwap = endian === ">" || (endian === "=" && !isLittleEndian());
    if (needsByteSwap && itemSize > 1) data = byteSwap(data, itemSize, TypedArray);

    return { data, shape, dtype: descr };
}

function isLittleEndian() {
    const buffer = new ArrayBuffer(2);
    new DataView(buffer).setInt16(0, 1, true);
    return new Int16Array(buffer)[0] === 1;
}

function byteSwap(array, itemSize, TypedArray) {
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    const swapped = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i += itemSize) {
        for (let j = 0; j < itemSize; j++) swapped[i + j] = bytes[i + itemSize - 1 - j];
    }
    return new TypedArray(swapped.buffer, swapped.byteOffset, swapped.byteLength / itemSize);
}

function normalizeName(name) {
    return name.replace(/^.*\//, "").replace(/\.npy$/i, "");
}

/**
 * Loads a packed .npz OR a plain zip of loose .npy files (same thing to a zip reader).
 * A single unparseable entry becomes a warning in `arrays.__warnings`, not a crash.
 */
export async function loadNPZ(source) {
    let buffer;
    if (source instanceof ArrayBuffer) buffer = source;
    else if (source instanceof Blob) buffer = await source.arrayBuffer();
    else if (typeof source === "string") {
        const response = await fetch(source);
        if (!response.ok) {
            throw new Error(`Could not load analysis file: ${response.status} ${response.statusText}`);
        }
        buffer = await response.arrayBuffer();
    } else throw new Error("Unsupported NPZ source");

    const head = new Uint8Array(buffer, 0, Math.min(4, buffer.byteLength));
    if (head[0] !== 0x50 || head[1] !== 0x4b) {
        const preview = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(60, buffer.byteLength)));
        throw new Error(
            `Not a zip/npz file (${buffer.byteLength} bytes, starts with "${preview.replace(/\s+/g, " ").trim()}"). ` +
            `If this looks like HTML, the URL is wrong or the file is missing from public/.`
        );
    }

    const zipped = unzipSync(new Uint8Array(buffer));
    const arrays = {};
    const warnings = [];

    for (const [filename, bytes] of Object.entries(zipped)) {
        if (!filename.toLowerCase().endsWith(".npy")) continue;
        try {
            arrays[normalizeName(filename)] = parseNpy(bytes);
        } catch (err) {
            warnings.push(`${filename}: ${err.message}`);
            console.warn(`Skipped ${filename}:`, err.message);
        }
    }

    if (Object.keys(arrays).length === 0) throw new Error("NPZ file contains no readable NPY arrays");

    Object.defineProperty(arrays, "__warnings", { value: warnings, enumerable: false });
    return arrays;
}

export function getArray(arrays, names) {
    for (const name of names) if (arrays[name]) return arrays[name];

    const normalized = Object.keys(arrays).map((key) => ({ original: key, lower: key.toLowerCase() }));
    for (const name of names) {
        const match = normalized.find(
            (e) => e.lower === name.toLowerCase() || e.lower.includes(name.toLowerCase())
        );
        if (match) return arrays[match.original];
    }
    return null;
}

/** 0-d numeric array -> number (or null) */
export function getScalar(arrays, names) {
    const arr = getArray(arrays, names);
    if (!arr || arr.isString || !arr.data?.length) return null;
    const n = Number(arr.data[0]);
    return Number.isFinite(n) ? n : null;
}

/** 0-d string array -> string (or null) */
export function getString(arrays, names) {
    const arr = getArray(arrays, names);
    return arr?.isString && arr.data.length ? arr.data[0] : null;
}

/**
 * metrics_before / metrics_after are JSON strings inside the zip ("null" when absent).
 * Returns { before, after } where each is { pooled:{rmse,mae,correlation,n}, perClass, baseline } or null.
 */
export function getMetrics(arrays) {
    const parse = (names) => {
        const text = getString(arrays, names);
        if (!text) return null;
        try {
            const obj = JSON.parse(text);
            if (!obj || !obj.pooled) return null;
            return {
                pooled: obj.pooled,
                perClass: obj.per_class ?? null,
                baseline: obj.baseline_class_mean ?? null,
                nClamped: obj.n_clamped ?? null
            };
        } catch {
            return null;
        }
    };
    const before = parse(["metrics_before"]);
    const after = parse(["metrics_after"]);
    return before || after ? { before, after } : null;
}

/** Optional-field aware view of the analysis zip. Before-fields may be null (1-GLB case). */
export function getAnalysis(arrays) {
    return {
        mode: getString(arrays, ["mode"]),
        noiseFloor: getScalar(arrays, ["noise_floor"]),
        elevationBefore: getArray(arrays, ["elevation_before"]),
        elevationAfter: getArray(arrays, ["elevation_after"]),
        elevationChange: getArray(arrays, ["elevation_change"]),
        confidenceBefore: getArray(arrays, ["confidence_before"]),
        confidenceAfter: getArray(arrays, ["confidence_after"]),
        combinedConfidence: getArray(arrays, ["combined_confidence"]),
        classChanged: getArray(arrays, ["class_changed"]),
        significantChange: getArray(arrays, ["significant_change"]),
        metrics: getMetrics(arrays),
        warnings: arrays.__warnings ?? []
    };
}

export function arrayStats(array) {
    if (!array?.data?.length || array.isString) return null;
    let min = Infinity, max = -Infinity;
    for (const value of array.data) {
        const n = Number(value);
        if (!Number.isFinite(n)) continue;
        if (n < min) min = n;
        if (n > max) max = n;
    }
    if (min === Infinity) return null;
    return { min, max, count: array.data.length, shape: array.shape, dtype: array.dtype };
}