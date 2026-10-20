import { useState, useEffect, useRef } from 'react';
import { Network, Play, Loader2, Maximize2, Minimize2, ZoomIn, ZoomOut, RotateCcw, Info } from 'lucide-react';
import api from '../api/client';
import useUploadStore from '../store/uploadStore';
import DrugAdrNetworkSigma from '../components/DrugAdrNetworkSigma';

const AGE_GROUPS = ['NEONATE', 'INFANT', 'CHILD', 'ADOLESCENT'];

export default function GNNNetworkView() {
  const [ageGroup, setAgeGroup] = useState('CHILD');
  const [graphData, setGraphData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [trainingStatus, setTrainingStatus] = useState({});
  const [viewMode, setViewMode] = useState('classical');
  const [loading, setLoading] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const sigmaRef = useRef(null);
  const { selectedYear, selectedQuarter, getActiveQuarterFilter } = useUploadStore();

  useEffect(() => { loadGraphData(); loadStatus(); }, [ageGroup, selectedYear, selectedQuarter]);

  const loadGraphData = async () => {
    setLoading(true);
    try {
      const qFilter = getActiveQuarterFilter();
      const res = await api.get(`/gnn/graph-data/${ageGroup}?quarter=${qFilter}`);
      setGraphData(res.data);
    } catch (err) { console.error(err); }
    setLoading(false);
  };

  const loadStatus = async () => {
    try { 
      const qFilter = getActiveQuarterFilter();
      const res = await api.get(`/gnn/status?quarter=${qFilter}`); 
      setTrainingStatus(res.data); 
    } catch {}
  };

  const handleTrain = async () => {
    try {
      const qFilter = getActiveQuarterFilter();
      const res = await api.post(`/gnn/train/${ageGroup}?epochs=100&quarter=${qFilter}`);
      setTrainingStatus(p => ({ ...p, [ageGroup]: res.data }));
      const predRes = await api.get(`/gnn/predictions/${ageGroup}?top_k=30&quarter=${qFilter}`);
      setPredictions(predRes.data.predictions || []);
    } catch (err) { console.error(err); }
  };

  const agStatus = trainingStatus[ageGroup];
  const svgH = isFullscreen ? 'calc(100vh - 130px)' : '600px';
  const nodes = graphData?.nodes || [];
  const edges = graphData?.edges || [];
  const drugCount = nodes.filter(n => n.type === 'drug').length;
  const adrCount  = nodes.filter(n => n.type === 'adr').length;
  const signalCount = edges.filter(e => e.is_signal).length;

  const graphPanel = (
    <div className="relative" style={{ height: svgH }}>
      <DrugAdrNetworkSigma ref={sigmaRef} graphData={graphData} height="100%" />

      {/* Controls overlay */}
      <div className="absolute top-3 right-3 flex flex-col gap-1.5 z-10">
        <button onClick={() => setIsFullscreen(f => !f)} className="p-2 bg-surface-dark/90 border border-surface-border rounded-lg hover:bg-pulse-600/20" title="Fullscreen">
          {isFullscreen ? <Minimize2 className="w-4 h-4 text-gray-300" /> : <Maximize2 className="w-4 h-4 text-gray-300" />}
        </button>
        <button onClick={() => sigmaRef.current?.zoomIn()} className="p-2 bg-surface-dark/90 border border-surface-border rounded-lg hover:bg-pulse-600/20"><ZoomIn className="w-4 h-4 text-gray-300" /></button>
        <button onClick={() => sigmaRef.current?.zoomOut()} className="p-2 bg-surface-dark/90 border border-surface-border rounded-lg hover:bg-pulse-600/20"><ZoomOut className="w-4 h-4 text-gray-300" /></button>
        <button onClick={() => sigmaRef.current?.resetView()} className="p-2 bg-surface-dark/90 border border-surface-border rounded-lg hover:bg-pulse-600/20"><RotateCcw className="w-4 h-4 text-gray-300" /></button>
      </div>
    </div>
  );

  if (isFullscreen) {
    return (
      <div className="fixed inset-0 z-50 bg-[#0a0a19] flex flex-col">
        <div className="flex items-center justify-between px-4 py-2 bg-surface-dark border-b border-surface-border">
          <div className="flex items-center gap-4">
            <h2 className="text-sm font-semibold text-gray-200">Drug–ADR Network ({ageGroup})</h2>
            <select value={ageGroup} onChange={e => setAgeGroup(e.target.value)} className="select-field w-36 text-xs">
              {AGE_GROUPS.map(ag => <option key={ag} value={ag}>{ag}</option>)}
            </select>
          </div>
          <button onClick={() => setIsFullscreen(false)} className="p-2 hover:bg-surface-hover rounded-lg"><Minimize2 className="w-5 h-5 text-gray-300" /></button>
        </div>
        <div className="flex-1 overflow-hidden p-2">{graphPanel}</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <div style={{display:'flex', gap:'8px', marginBottom:'12px'}}>
            <span style={{fontSize:'12px', color:'#9CA3AF'}}>
              Showing:
            </span>
            <span style={{fontSize:'12px', fontFamily:'monospace', 
                          background:'#1F2937',
                          padding:'2px 8px', borderRadius:'4px', color:'#93C5FD'}}>
              {selectedYear === 'ALL' ? 'All Years' : selectedYear}
              {' / '}
              {selectedQuarter === 'ALL'
                ? (selectedYear === 'ALL' ? 'All Quarters' : `All ${selectedYear}`)
                : selectedQuarter}
            </span>
          </div>
          <h1 className="text-2xl font-bold gradient-text">GNN Network View</h1>
          <p className="text-sm text-gray-500 mt-1">Heterogeneous Attention Network — Drug-ADR link prediction</p>
        </div>
        <div className="flex items-center gap-3">
          <select value={ageGroup} onChange={e => setAgeGroup(e.target.value)} className="select-field w-40" id="gnn-age-select">
            {AGE_GROUPS.map(ag => <option key={ag} value={ag}>{ag}</option>)}
          </select>
          <div className="flex bg-surface-dark rounded-lg border border-surface-border p-0.5">
            {['classical', 'gnn', 'both'].map(m => (
              <button key={m} onClick={() => setViewMode(m)}
                className={`px-3 py-1.5 text-xs rounded-md transition-all ${viewMode === m ? 'bg-pulse-600 text-white' : 'text-gray-400 hover:text-gray-200'}`}>
                {m === 'classical' ? 'Classical' : m === 'gnn' ? 'GNN Predicted' : 'Both'}
              </button>
            ))}
          </div>
          <button onClick={handleTrain} className="btn-primary flex items-center gap-2" id="train-gnn-btn">
            <Play className="w-4 h-4" /> Train GNN
          </button>
        </div>
      </div>

      {agStatus && (
        <div className={`glass-card p-4 ${agStatus.status === 'completed' ? 'border-neon-green/30' : ''}`}>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-300">
              {agStatus.status === 'completed' ? '✓ Training Complete' : `Status: ${agStatus.status}`}
              {agStatus.status === 'training' && ` — Epoch ${agStatus.epoch}/${agStatus.total_epochs}`}
            </span>
            {agStatus.status === 'completed' && (
              <div className="flex gap-4 text-xs text-gray-400">
                <span>AUC: <span className="text-neon-green font-mono">{agStatus.test_auc}</span></span>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-card p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-300">Drug–ADR Network ({ageGroup})</h3>
            <div className="flex items-center gap-1 text-[10px] text-gray-500"><Info className="w-3 h-3" /> Hover · Scroll · Drag</div>
          </div>
          {loading ? (
            <div className="flex items-center justify-center h-[600px]"><Loader2 className="w-10 h-10 animate-spin text-pulse-500" /></div>
          ) : (!graphData || !nodes.length) ? (
            <div className="flex items-center justify-center h-[600px] text-gray-500">
              <div className="text-center"><Network className="w-12 h-12 mx-auto mb-3 opacity-30" /><p>No data</p></div>
            </div>
          ) : graphPanel}
        </div>

        <div className="space-y-4">
          <div className="glass-card p-4">
            <h3 className="text-sm font-semibold text-gray-300 mb-3">GNN Novel Predictions</h3>
            <div className="space-y-2 max-h-[400px] overflow-y-auto">
              {predictions.length === 0 ? (
                <p className="text-xs text-gray-500 py-4 text-center">Train GNN to see predictions</p>
              ) : predictions.slice(0, 20).map((p, i) => (
                <div key={i} className="p-2 bg-surface-dark/50 rounded-lg border border-surface-border/50 hover:border-neon-purple/30 transition-colors">
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-xs text-neon-cyan truncate" title={p.drug}>{p.drug}</span>
                    <span className="text-[10px] text-gray-600">→</span>
                    <span className="text-xs text-neon-amber truncate" title={p.adr}>{p.adr}</span>
                  </div>
                  <div className="flex items-center justify-between mt-1">
                    <div className="w-16 h-1 bg-surface-border rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-neon-purple to-neon-pink rounded-full" style={{ width: `${p.confidence * 100}%` }} />
                    </div>
                    <span className="text-[10px] font-mono text-neon-purple">{(p.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {nodes.length > 0 && (
            <div className="glass-card p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">Network Summary</h3>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-surface-dark/50 p-2 rounded border border-surface-border/40">
                  <div className="text-gray-500">Drug Nodes</div>
                  <div className="text-lg font-bold text-pulse-400">{drugCount}</div>
                </div>
                <div className="bg-surface-dark/50 p-2 rounded border border-surface-border/40">
                  <div className="text-gray-500">ADR Nodes</div>
                  <div className="text-lg font-bold text-neon-amber">{adrCount}</div>
                </div>
                <div className="bg-surface-dark/50 p-2 rounded border border-surface-border/40">
                  <div className="text-gray-500">Total Edges</div>
                  <div className="text-lg font-bold text-gray-300">{edges.length}</div>
                </div>
                <div className="bg-surface-dark/50 p-2 rounded border border-surface-border/40">
                  <div className="text-gray-500">Signal Edges</div>
                  <div className="text-lg font-bold text-red-400">{signalCount}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
