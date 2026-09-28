import { unzipSync } from "fflate";

function readAscii(bytes, start, end) {
    return new TextDecoder("latin1").decode(bytes.slice(start, end));
}

function parseNpy(buffer) {
    const bytes = new Uint8Array(buffer);

    if (
        bytes[0] !== 0x93 ||
        bytes[1] !== 0x4e ||
        bytes[2] !== 0x55 ||
        bytes[3] !== 0x4d ||
        bytes[4] !== 0x50 ||
        bytes[5] !== 0x59
    ) {
        throw new Error("Invalid NPY file");
    }

    const major = bytes[6];
    const minor = bytes[7];

    let headerLength;
    let headerStart;

    if (major === 1) {
        headerLength = bytes[8] | (bytes[9] << 8);
        headerStart = 10;
    } else if (major === 2 || major === 3) {
        headerLength =
            bytes[8] |
            (bytes[9] << 8) |
            (bytes[10] << 16) |
            (bytes[11] << 24);

        headerStart = 12;
    } else {
        throw new Error(`Unsupported NPY version ${major}.${minor}`);
    }

    const header = readAscii(
        bytes,
        headerStart,
        headerStart + headerLength
    );

    const descrMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
    const shapeMatch = header.match(/'shape'\s*:\s*\(([^)]*)\)/);
    const fortranMatch = header.match(/'fortran_order'\s*:\s*(True|False)/);

    if (!descrMatch || !shapeMatch) {
        throw new Error("Could not read NPY header");
    }

    const descr = descrMatch[1];

    const shapeText = shapeMatch[1].trim();

    const shape =
        shapeText.length === 0
            ? []
            : shapeText
                  .split(",")
                  .map((value) => value.trim())
                  .filter(Boolean)
                  .map(Number);

    const fortranOrder = fortranMatch
        ? fortranMatch[1] === "True"
        : false;

    if (fortranOrder) {
        throw new Error("Fortran-order NPY arrays are not supported");
    }

    const dataStart = headerStart + headerLength;
    const dataBytes = bytes.slice(dataStart);

    const endian = descr[0];
    const kind = descr[1];
    const itemSize = Number(descr.slice(2));

    let TypedArray;

    if (kind === "f" && itemSize === 4) {
        TypedArray = Float32Array;
    } else if (kind === "f" && itemSize === 8) {
        TypedArray = Float64Array;
    } else if (kind === "i" && itemSize === 1) {
        TypedArray = Int8Array;
    } else if (kind === "i" && itemSize === 2) {
        TypedArray = Int16Array;
    } else if (kind === "i" && itemSize === 4) {
        TypedArray = Int32Array;
    } else if (kind === "i" && itemSize === 8) {
        TypedArray = BigInt64Array;
    } else if (kind === "u" && itemSize === 1) {
        TypedArray = Uint8Array;
    } else if (kind === "u" && itemSize === 2) {
        TypedArray = Uint16Array;
    } else if (kind === "u" && itemSize === 4) {
        TypedArray = Uint32Array;
    } else if (kind === "u" && itemSize === 8) {
        TypedArray = BigUint64Array;
    } else if (kind === "b" && itemSize === 1) {
        TypedArray = Uint8Array;
    } else {
        throw new Error(`Unsupported NPY dtype: ${descr}`);
    }

    if (dataBytes.byteLength % itemSize !== 0) {
        throw new Error(`Invalid NPY data length for dtype ${descr}`);
    }

    const raw = dataBytes.buffer.slice(
        dataBytes.byteOffset,
        dataBytes.byteOffset + dataBytes.byteLength
    );

    let data = new TypedArray(raw);

    const needsByteSwap =
        endian === ">" ||
        (endian === "=" && !isLittleEndian());

    if (needsByteSwap && itemSize > 1) {
        data = byteSwap(data, itemSize, TypedArray);
    }

    return {
        data,
        shape,
        dtype: descr
    };
}

function isLittleEndian() {
    const buffer = new ArrayBuffer(2);
    new DataView(buffer).setInt16(0, 1, true);
    return new Int16Array(buffer)[0] === 1;
}

function byteSwap(array, itemSize, TypedArray) {
    const bytes = new Uint8Array(
        array.buffer,
        array.byteOffset,
        array.byteLength
    );

    const swapped = new Uint8Array(bytes.length);

    for (let i = 0; i < bytes.length; i += itemSize) {
        for (let j = 0; j < itemSize; j++) {
            swapped[i + j] = bytes[i + itemSize - 1 - j];
        }
    }

    return new TypedArray(
        swapped.buffer,
        swapped.byteOffset,
        swapped.byteLength / itemSize
    );
}

function normalizeName(name) {
    return name.replace(/\.npy$/i, "");
}

export async function loadNPZ(source) {
    let buffer;

    if (source instanceof ArrayBuffer) {
        buffer = source;
    } else if (source instanceof Blob) {
        buffer = await source.arrayBuffer();
    } else if (typeof source === "string") {
        const response = await fetch(source);

        if (!response.ok) {
            throw new Error(
                `Could not load analysis file: ${response.status} ${response.statusText}`
            );
        }

        buffer = await response.arrayBuffer();
    } else {
        throw new Error("Unsupported NPZ source");
    }

    const zipped = unzipSync(new Uint8Array(buffer));

    const arrays = {};

    for (const [filename, bytes] of Object.entries(zipped)) {
        if (!filename.toLowerCase().endsWith(".npy")) {
            continue;
        }

        const parsed = parseNpy(bytes.buffer);

        arrays[normalizeName(filename)] = parsed;
    }

    if (Object.keys(arrays).length === 0) {
        throw new Error("NPZ file contains no NPY arrays");
    }

    return arrays;
}

export function getArray(arrays, names) {
    for (const name of names) {
        if (arrays[name]) {
            return arrays[name];
        }
    }

    const normalized = Object.keys(arrays).map((key) => ({
        original: key,
        lower: key.toLowerCase()
    }));

    for (const name of names) {
        const match = normalized.find(
            (entry) =>
                entry.lower === name.toLowerCase() ||
                entry.lower.includes(name.toLowerCase())
        );

        if (match) {
            return arrays[match.original];
        }
    }

    return null;
}

export function arrayStats(array) {
    if (!array?.data?.length) {
        return null;
    }

    let min = Infinity;
    let max = -Infinity;

    for (const value of array.data) {
        const number = Number(value);

        if (!Number.isFinite(number)) {
            continue;
        }

        if (number < min) min = number;
        if (number > max) max = number;
    }

    if (min === Infinity || max === -Infinity) {
        return null;
    }

    return {
        min,
        max,
        count: array.data.length,
        shape: array.shape,
        dtype: array.dtype
    };
}