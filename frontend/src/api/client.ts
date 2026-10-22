import axios from 'axios'
export const api=axios.create({baseURL:import.meta.env.VITE_API_URL||'http://127.0.0.1:8000',timeout:30000})
export const getHealth=()=>api.get('/health').then(r=>r.data)
export const getModelInfo=()=>api.get('/model-info').then(r=>r.data)
export const getEvaluation=()=>api.get('/evaluation').then(r=>r.data)
export const predict=(p:Record<string,unknown>)=>api.post('/predict',p).then(r=>r.data)
export const predictSequence=(observations:Record<string,unknown>[])=>api.post('/predict-sequence',{observations}).then(r=>r.data)
export const explain=(p:Record<string,unknown>)=>api.post('/explain',p).then(r=>r.data)
export const validateDataset=(file:File)=>{const f=new FormData();f.append('file',file);return api.post('/validate-dataset',f).then(r=>r.data)}
