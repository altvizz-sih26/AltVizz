import { DEMO_CONFIG } from '../config/demoConfig';

const STATE_KEY = 'depthwizard-demo-job';
export const getJob = () => JSON.parse(sessionStorage.getItem(STATE_KEY) || 'null');
export const uploadImage = async (file) => {
  const job = { fileName: file.name, fileType: file.type || `image/${file.name.split('.').pop()}`, fileSize: file.size, previewUrl: file.type.startsWith('image/') && !/tiff/i.test(file.name) ? URL.createObjectURL(file) : null, demoMode: true, createdAt: Date.now() };
  sessionStorage.setItem(STATE_KEY, JSON.stringify(job)); return job;
};
export const startProcessing = async () => { const job = getJob(); if (!job) throw new Error('Select an image first.'); return job; };
export const getResults = async () => ({ ...(getJob() || { fileName: DEMO_CONFIG.defaultFileName, fileType: 'image/tiff' }), dsmName: DEMO_CONFIG.demoDsmName, glbName: DEMO_CONFIG.demoGlbName, status: 'Complete' });
export const clearDemoJob = () => sessionStorage.removeItem(STATE_KEY);

// A valid, minimal glTF 2.0 binary containing one triangle. Replace this with a backend URL later.
export const createDemoGlb = () => {
  const json = JSON.stringify({asset:{version:'2.0',generator:'DepthWizard demo'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{primitives:[{attributes:{POSITION:0},indices:1}]}],buffers:[{byteLength:42}],bufferViews:[{buffer:0,byteOffset:0,byteLength:36,target:34962},{buffer:0,byteOffset:36,byteLength:6,target:34963}],accessors:[{bufferView:0,componentType:5126,count:3,type:'VEC3',min:[-1,-1,0],max:[1,1,0]},{bufferView:1,componentType:5123,count:3,type:'SCALAR'}]});
  const enc = new TextEncoder(), jsonBytes = enc.encode(json), paddedJson = new Uint8Array(Math.ceil(jsonBytes.length / 4) * 4); paddedJson.set(jsonBytes); paddedJson.fill(32, jsonBytes.length);
  const bin = new ArrayBuffer(44), dv = new DataView(bin); new Float32Array(bin,0,9).set([-1,-1,0, 1,-1,0, 0,1,0]); dv.setUint16(36,0,true);dv.setUint16(38,1,true);dv.setUint16(40,2,true);
  const total = 12 + 8 + paddedJson.length + 8 + bin.byteLength, out = new ArrayBuffer(total), view = new DataView(out); view.setUint32(0,0x46546C67,true);view.setUint32(4,2,true);view.setUint32(8,total,true);view.setUint32(12,paddedJson.length,true);view.setUint32(16,0x4E4F534A,true);new Uint8Array(out,20,paddedJson.length).set(paddedJson);let o=20+paddedJson.length;view.setUint32(o,bin.byteLength,true);view.setUint32(o+4,0x004E4942,true);new Uint8Array(out,o+8).set(new Uint8Array(bin)); return new Blob([out],{type:'model/gltf-binary'});
};
export const downloadResult = () => { const link = document.createElement('a'); link.href = URL.createObjectURL(createDemoGlb()); link.download = DEMO_CONFIG.demoGlbName; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 2000); };
