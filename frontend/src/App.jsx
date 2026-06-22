import React, { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [modelInfo, setModelInfo] = useState({ loaded: false });
  const [loading, setLoading] = useState(true);
  const [predictLoading, setPredictLoading] = useState(false);
  const [inputs, setInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  
  // Batch prediction states
  const [batchFile, setBatchFile] = useState(null);
  const [batchPreview, setBatchPreview] = useState([]);
  const [batchLoading, setBatchLoading] = useState(false);
  
  // Model upload states
  const [modelFile, setModelFile] = useState(null);
  const [metadataFile, setMetadataFile] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  
  // UI Tabs and Notifications
  const [activeTab, setActiveTab] = useState('predict');
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  };

  const fetchModelInfo = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/model-info`);
      if (!res.ok) throw new Error('Failed to fetch model info');
      const data = await res.json();
      setModelInfo(data);
      
      // Initialize inputs with default values from metadata
      if (data.loaded && data.features) {
        const initialInputs = {};
        data.features.forEach(f => {
          if (f.type === 'numerical') {
            initialInputs[f.name] = f.default !== undefined ? f.default : (f.min !== undefined ? f.min : 0);
          } else if (f.type === 'categorical') {
            initialInputs[f.name] = f.options && f.options.length > 0 ? f.options[0] : '';
          } else {
            initialInputs[f.name] = '';
          }
        });
        setInputs(initialInputs);
      }
    } catch (err) {
      console.error(err);
      showToast('Could not connect to the backend server.', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const handleInputChange = (name, value) => {
    setInputs(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setPredictLoading(true);
    setPredictionResult(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: inputs })
      });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }
      const data = await res.json();
      setPredictionResult(data);
      showToast('Inference completed successfully!');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setPredictLoading(false);
    }
  };

  const handleFileChange = (e, fileType) => {
    const file = e.target.files[0];
    if (fileType === 'model') {
      setModelFile(file);
    } else {
      setMetadataFile(file);
    }
  };

  const handleUploadModel = async (e) => {
    e.preventDefault();
    if (!modelFile || !metadataFile) {
      showToast('Please select both model.joblib and metadata.json files.', 'error');
      return;
    }

    setUploadLoading(true);
    const formData = new FormData();
    formData.append('model', modelFile);
    formData.append('metadata', metadataFile);

    try {
      const res = await fetch(`${API_BASE_URL}/api/upload-model`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to upload files.');
      }
      const data = await res.json();
      showToast(data.message || 'Model loaded successfully!');
      setModelFile(null);
      setMetadataFile(null);
      // Reset input files
      document.getElementById('model-upload-input').value = '';
      document.getElementById('meta-upload-input').value = '';
      
      // Reload details and switch to predict tab
      await fetchModelInfo();
      setActiveTab('predict');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setUploadLoading(false);
    }
  };

  const handleDeleteModel = async () => {
    if (!window.confirm('Are you sure you want to delete the model artifacts?')) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/delete-model`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Deletion failed.');
      showToast('Model artifacts deleted.');
      setModelInfo({ loaded: false });
      setInputs({});
      setPredictionResult(null);
      setBatchFile(null);
      setBatchPreview([]);
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Batch Prediction handling
  const handleBatchFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setBatchFile(file);
    
    // Read a simple preview
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const rows = text.split('\n').filter(r => r.trim() !== '');
      const previewRows = rows.slice(0, 5).map(row => row.split(','));
      setBatchPreview(previewRows);
    };
    reader.readAsText(file);
  };

  const handleBatchPredict = async (e) => {
    e.preventDefault();
    if (!batchFile) return;
    setBatchLoading(true);
    
    const formData = new FormData();
    formData.append('file', batchFile);
    
    try {
      const res = await fetch(`${API_BASE_URL}/api/predict-batch`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Batch prediction failed');
      }
      
      // Process download of predictions CSV
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `predictions_${batchFile.name}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      
      showToast('Batch prediction download triggered!');
      setBatchFile(null);
      setBatchPreview([]);
      document.getElementById('batch-file-input').value = '';
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setBatchLoading(false);
    }
  };

  // Helper to determine active confidence label/probability
  const getPredictionConfidence = () => {
    if (!predictionResult || !predictionResult.probabilities) return null;
    const pred = predictionResult.prediction;
    
    // Support matching prediction names either raw or label-resolved
    let prob = predictionResult.probabilities[pred];
    
    // Fallback if class names in keys are different case
    if (prob === undefined) {
      const key = Object.keys(predictionResult.probabilities).find(
        k => k.toLowerCase() === pred.toLowerCase()
      );
      if (key) prob = predictionResult.probabilities[key];
    }
    
    return prob !== undefined ? Math.round(prob * 100) : null;
  };

  // Flower Image dynamic lookup helper
  const getFlowerImage = () => {
    if (!predictionResult || !predictionResult.prediction) return null;
    const pred = predictionResult.prediction.toLowerCase();
    
    if (pred.includes('setosa')) {
      return '/images/setosa.png';
    } else if (pred.includes('versicolor')) {
      return '/images/versicolor.png';
    } else if (pred.includes('virginica')) {
      return '/images/virginica.png';
    }
    return null;
  };

  const flowerImageUrl = getFlowerImage();

  return (
    <div className="app-container">
      {/* Toast Notification */}
      {toast && (
        <div className={`toast ${toast.type}`}>
          <div className="pulse-dot"></div>
          <span>{toast.message}</span>
        </div>
      )}

      {/* Header */}
      <header className="app-header">
        <div className="logo-section">
          <div className="logo-icon">B</div>
          <div className="logo-text">
            <h1>OMNICLASSIFIER TERMINAL</h1>
            <p>ADAPTIVE MACHINE LEARNING SERVICE</p>
          </div>
        </div>
        
        <div className={`status-badge ${modelInfo.loaded ? 'online' : 'no-model'}`}>
          <div className="pulse-dot"></div>
          <span>{modelInfo.loaded ? 'ONLINE' : 'PENDING'}</span>
        </div>
      </header>

      {/* Loading Cover State */}
      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '100px 0', gap: '20px', border: '1px solid var(--border-color)' }}>
          <div className="spinner"></div>
          <p style={{ color: 'var(--color-primary)' }}>SERVICE INITIALIZATION IN PROGRESS...</p>
        </div>
      ) : (
        <div className="dashboard-grid">
          
          {/* SIDEBAR: Model Stats / Actions */}
          <aside className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="card-title-container" style={{ marginBottom: '0.25rem' }}>
              <div className="card-title">
                [SYSTEM STATS]
              </div>
            </div>

            {modelInfo.loaded ? (
              <div className="model-info-list">
                <div className="info-item">
                  <span className="info-label">IDENT</span>
                  <span className="info-value">{modelInfo.model_name}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">PIPELINE</span>
                  <span className="info-value badge">{modelInfo.model_type}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">TARGET</span>
                  <span className="info-value" style={{ color: 'var(--color-primary)' }}>
                    {modelInfo.target}
                  </span>
                </div>
                
                {modelInfo.metrics && Object.keys(modelInfo.metrics).length > 0 && (
                  <div>
                    <h3 className="info-label" style={{ marginBottom: '0.5rem', marginTop: '0.25rem' }}>METRICS REPORT</h3>
                    <div className="metrics-grid">
                      {Object.entries(modelInfo.metrics).map(([key, val]) => (
                        <div key={key} className="metric-widget">
                          <span className="metric-widget-value">
                            {typeof val === 'number' ? (val <= 1 ? `${Math.round(val * 100)}%` : val.toFixed(2)) : val}
                          </span>
                          <span className="metric-widget-label">{key.replace('_', ' ').toUpperCase()}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <button 
                  onClick={handleDeleteModel} 
                  className="btn btn-danger" 
                  style={{ marginTop: '0.5rem', width: '100%' }}
                >
                  DE-REGISTER ARTIFACTS
                </button>
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textAlign: 'center', padding: '20px 0' }}>
                <p>NO CLASSIFICATION PIPELINE REGISTERED.</p>
                <p style={{ marginTop: '10px' }}>RUN NOTEBOOK WORKFLOW OR UPLOAD SCHEMAS.</p>
              </div>
            )}
          </aside>

          {/* MAIN PANELS: Interactive Predictions, Uploader and Charts */}
          <main style={{ minWidth: 0 }}>
            {modelInfo.loaded ? (
              <>
                <nav className="tab-navigation">
                  <button 
                    className={`nav-tab ${activeTab === 'predict' ? 'active' : ''}`}
                    onClick={() => setActiveTab('predict')}
                  >
                    INFERENCE
                  </button>
                  <button 
                    className={`nav-tab ${activeTab === 'charts' ? 'active' : ''}`}
                    onClick={() => setActiveTab('charts')}
                  >
                    PERFORMANCE CHARTS
                  </button>
                  <button 
                    className={`nav-tab ${activeTab === 'batch' ? 'active' : ''}`}
                    onClick={() => setActiveTab('batch')}
                  >
                    BATCH (CSV)
                  </button>
                  <button 
                    className={`nav-tab ${activeTab === 'upload' ? 'active' : ''}`}
                    onClick={() => setActiveTab('upload')}
                  >
                    DEPLOY
                  </button>
                </nav>

                {/* TAB 1: Single Prediction */}
                {activeTab === 'predict' && (
                  <section className="glass-card">
                    <div className="card-title-container">
                      <div className="card-title">[RUN PARAMETERS]</div>
                    </div>
                    <form onSubmit={handlePredict} className="prediction-form">
                      {modelInfo.features && modelInfo.features.map(f => {
                        const isText = f.type === 'text';
                        const isCategorical = f.type === 'categorical';
                        
                        return (
                          <div 
                            key={f.name} 
                            className={`form-group ${isText ? 'full-width-form-item' : ''}`}
                          >
                            <div className="form-label-container">
                              <label className="form-label">
                                {f.name.toUpperCase()}
                                {f.type === 'numerical' && (
                                  <span className="form-val-display">{inputs[f.name]}</span>
                                )}
                              </label>
                            </div>
                            
                            {f.description && <span className="form-desc">{f.description.toUpperCase()}</span>}

                            {isText ? (
                              <textarea
                                className="form-textarea"
                                rows="3"
                                value={inputs[f.name] || ''}
                                onChange={(e) => handleInputChange(f.name, e.target.value)}
                                placeholder="ENTER INPUT CHARACTER DATA..."
                                required
                              />
                            ) : isCategorical ? (
                              <select
                                className="form-select"
                                value={inputs[f.name] || ''}
                                onChange={(e) => handleInputChange(f.name, e.target.value)}
                                required
                              >
                                {f.options && f.options.map(opt => (
                                  <option key={opt} value={opt}>{opt}</option>
                                ))}
                              </select>
                            ) : (
                              // Numerical Range or input box
                              <div className="slider-container">
                                <input
                                  type="range"
                                  className="form-slider"
                                  min={f.min !== undefined ? f.min : 0}
                                  max={f.max !== undefined ? f.max : 100}
                                  step={f.step !== undefined ? f.step : 1}
                                  value={inputs[f.name] !== undefined ? inputs[f.name] : (f.min || 0)}
                                  onChange={(e) => handleInputChange(f.name, parseFloat(e.target.value))}
                                />
                                <input
                                  type="number"
                                  className="form-input"
                                  style={{ width: '90px', color: 'var(--color-secondary)', fontWeight: 'bold' }}
                                  min={f.min !== undefined ? f.min : 0}
                                  max={f.max !== undefined ? f.max : 100}
                                  step={f.step !== undefined ? f.step : 1}
                                  value={inputs[f.name] !== undefined ? inputs[f.name] : (f.min || 0)}
                                  onChange={(e) => handleInputChange(f.name, e.target.value === '' ? '' : parseFloat(e.target.value))}
                                  required
                                />
                              </div>
                            )}
                          </div>
                        );
                      })}

                      <div className="full-width-form-item" style={{ marginTop: '0.5rem' }}>
                        <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={predictLoading}>
                          {predictLoading ? 'CALCULATING ENGINE OUTPUTS...' : 'RUN CLASSIFICATION'}
                        </button>
                      </div>
                    </form>

                    {/* Inference Result Output */}
                    {predictionResult && (
                      <div className="output-pane has-prediction">
                        <div className="result-hero" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div style={{ display: 'flex', gap: '2rem', alignItems: 'center', flexWrap: 'wrap' }}>
                            <div className="result-badge">
                              <span className="result-badge-label">PREDICTED CLASS</span>
                              <span className="result-badge-value">{predictionResult.prediction.toUpperCase()}</span>
                            </div>

                            {getPredictionConfidence() !== null && (
                              <div className="confidence-bar-container">
                                <span className="confidence-text">CONFIDENCE LEVEL</span>
                                <div className="confidence-track">
                                  <div 
                                    className="confidence-fill" 
                                    style={{ width: `${getPredictionConfidence()}%` }}
                                  ></div>
                                </div>
                                <span style={{ fontSize: '0.75rem', fontWeight: '700', alignSelf: 'flex-end', color: 'var(--color-secondary)' }}>
                                  {getPredictionConfidence()}% CONFIDENCE
                                </span>
                              </div>
                            )}
                          </div>

                          {/* DYNAMIC IMAGE OVERLAY */}
                          {flowerImageUrl && (
                            <div className="prediction-image-container">
                              <img 
                                src={flowerImageUrl} 
                                alt={predictionResult.prediction} 
                                className="prediction-image" 
                              />
                            </div>
                          )}
                        </div>

                        {/* Complete Probabilities Breakdown */}
                        {predictionResult.probabilities && Object.keys(predictionResult.probabilities).length > 0 && (
                          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                            <div className="prob-dist-title">PROBABILITY MATRIX DISTRIBUTION</div>
                            <div className="prob-list">
                              {Object.entries(predictionResult.probabilities).map(([className, probability]) => {
                                const isHighest = className.toLowerCase() === predictionResult.prediction.toLowerCase();
                                const widthVal = `${Math.round(probability * 100)}%`;
                                return (
                                  <div key={className} className="prob-row">
                                    <span className="prob-label" title={className}>{className.toUpperCase()}</span>
                                    <div className="prob-bar-track">
                                      <div 
                                        className={`prob-bar-fill ${isHighest ? 'highest' : ''}`}
                                        style={{ width: widthVal }}
                                      ></div>
                                    </div>
                                    <span className="prob-percentage">{widthVal}</span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </section>
                )}

                {/* TAB 2: Performance Charts */}
                {activeTab === 'charts' && (
                  <section className="glass-card">
                    <div className="card-title-container">
                      <div className="card-title">[MODEL VISUALIZATION GRAPHICS]</div>
                    </div>
                    <div className="charts-tab-container">
                      <div className="chart-card">
                        <div className="chart-header">TRAINING GRAPH: RF FEATURE IMPORTANCE</div>
                        <div className="chart-img-container">
                          <img 
                            src={`${API_BASE_URL}/api/charts/train_chart.png?t=${new Date().getTime()}`} 
                            alt="Feature Importance Chart" 
                            className="chart-img"
                            onError={(e) => {
                              e.target.style.display = 'none';
                              e.target.parentNode.innerHTML = '<div class="prediction-image-placeholder">TRAINING PLOT ARTIFACT NOT FOUND IN PATH</div>';
                            }}
                          />
                        </div>
                      </div>

                      <div className="chart-card">
                        <div className="chart-header">TESTING GRAPH: TEST CONFUSION MATRIX</div>
                        <div className="chart-img-container">
                          <img 
                            src={`${API_BASE_URL}/api/charts/test_chart.png?t=${new Date().getTime()}`} 
                            alt="Confusion Matrix Chart" 
                            className="chart-img"
                            onError={(e) => {
                              e.target.style.display = 'none';
                              e.target.parentNode.innerHTML = '<div class="prediction-image-placeholder">TESTING PLOT ARTIFACT NOT FOUND IN PATH</div>';
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </section>
                )}

                {/* TAB 3: Batch Predict */}
                {activeTab === 'batch' && (
                  <section className="glass-card">
                    <div className="card-title-container">
                      <div className="card-title">[BATCH INFERENCE UNIT]</div>
                    </div>

                    <form onSubmit={handleBatchPredict} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      <div className="upload-container" onClick={() => document.getElementById('batch-file-input').click()}>
                        <input
                          id="batch-file-input"
                          type="file"
                          accept=".csv"
                          style={{ display: 'none' }}
                          onChange={handleBatchFileChange}
                          required
                        />
                        <div className="upload-icon">F</div>
                        <div className="upload-text">
                          <h3>{batchFile ? batchFile.name.toUpperCase() : 'SELECT TARGET CSV FILE'}</h3>
                          <p>{batchFile ? `${(batchFile.size / 1024).toFixed(1)} KB` : 'CLICK TO MOUNT TARGET CSV SOURCE'}</p>
                        </div>
                      </div>

                      {batchPreview.length > 0 && (
                        <div>
                          <h4 style={{ fontSize: '0.8rem', fontWeight: '700', marginBottom: '0.25rem', color: 'var(--color-primary)' }}>HEADER PREVIEW</h4>
                          <div className="table-wrapper">
                            <table className="preview-table">
                              <thead>
                                <tr>
                                  {batchPreview[0].map((header, index) => (
                                    <th key={index}>{header.toUpperCase()}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {batchPreview.slice(1).map((row, rIndex) => (
                                  <tr key={rIndex}>
                                    {row.map((cell, cIndex) => (
                                      <td key={cIndex}>{cell}</td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}

                      <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={!batchFile || batchLoading}>
                        {batchLoading ? 'RUNNING BATCH PIPELINE...' : 'EXECUTE BATCH INFERENCE & DOWNLOAD'}
                      </button>
                    </form>
                  </section>
                )}

                {/* TAB 4: Update Model */}
                {activeTab === 'upload' && (
                  <section className="glass-card">
                    <div className="card-title-container">
                      <div className="card-title">[DEPLOY PIPELINE SYSTEM]</div>
                    </div>
                    <form onSubmit={handleUploadModel} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      <div className="dual-uploader">
                        <div className={`uploader-box ${modelFile ? 'file-selected' : ''}`}>
                          <h4>MODEL PIPELINE</h4>
                          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>SELECT model.joblib</p>
                          <input
                            id="model-upload-input"
                            type="file"
                            accept=".joblib,.pkl"
                            onChange={(e) => handleFileChange(e, 'model')}
                          />
                          {modelFile && <span className="file-name-display">{modelFile.name.toUpperCase()}</span>}
                        </div>

                        <div className={`uploader-box ${metadataFile ? 'file-selected' : ''}`}>
                          <h4>METADATA SCHEMA</h4>
                          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>SELECT metadata.json</p>
                          <input
                            id="meta-upload-input"
                            type="file"
                            accept=".json"
                            onChange={(e) => handleFileChange(e, 'metadata')}
                          />
                          {metadataFile && <span className="file-name-display">{metadataFile.name.toUpperCase()}</span>}
                        </div>
                      </div>

                      <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={uploadLoading}>
                        {uploadLoading ? 'DEPLOYING ARTIFACT FILES...' : 'MOUNT NEW ARTIFACTS'}
                      </button>
                    </form>
                  </section>
                )}
              </>
            ) : (
              /* NO MODEL PENDING VIEW */
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {/* Upload zone */}
                <section className="glass-card">
                  <div className="card-title-container">
                    <div className="card-title">[MOUNT ARTIFACT SOURCE]</div>
                  </div>

                  <form onSubmit={handleUploadModel} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div className="dual-uploader">
                      <div className={`uploader-box ${modelFile ? 'file-selected' : ''}`}>
                        <h4>MODEL PIPELINE</h4>
                        <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>SELECT model.joblib</p>
                        <input
                          id="model-upload-input"
                          type="file"
                          accept=".joblib,.pkl"
                          onChange={(e) => handleFileChange(e, 'model')}
                        />
                        {modelFile && <span className="file-name-display">{modelFile.name.toUpperCase()}</span>}
                      </div>

                      <div className={`uploader-box ${metadataFile ? 'file-selected' : ''}`}>
                        <h4>METADATA SCHEMA</h4>
                        <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>SELECT metadata.json</p>
                        <input
                          id="meta-upload-input"
                          type="file"
                          accept=".json"
                          onChange={(e) => handleFileChange(e, 'metadata')}
                        />
                        {metadataFile && <span className="file-name-display">{metadataFile.name.toUpperCase()}</span>}
                      </div>
                    </div>

                    <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={uploadLoading}>
                      {uploadLoading ? 'DEPLOYING ARTIFACT FILES...' : 'MOUNT SYSTEM ARTIFACTS'}
                    </button>
                  </form>
                </section>

                {/* Tutorial / Help */}
                <section className="glass-card">
                  <div className="card-title-container">
                    <div className="card-title">[INTEGRATION TUTORIAL]</div>
                  </div>
                  <div className="tutorial-section">
                    <div className="step-card">
                      <div className="step-number">1</div>
                      <div className="step-content">
                        <h4>OPEN model_development.ipynb</h4>
                        <p>LOCATED IN WORKSPACE ROOT. RUN PIPELINE CELLS TO FIT CLASSIFICATION ESTIMATOR.</p>
                      </div>
                    </div>

                    <div className="step-card">
                      <div className="step-number">2</div>
                      <div className="step-content">
                        <h4>EXPORT SYSTEM FILE BINARIES</h4>
                        <p>COMPILE THE PIPELINE AND WRITE OUT AS SPECIFIED:</p>
                        <code className="code-snippet">backend/artifacts/model.joblib</code>
                        <br />
                        <code className="code-snippet">backend/artifacts/metadata.json</code>
                      </div>
                    </div>

                    <div className="step-card">
                      <div className="step-number">3</div>
                      <div className="step-content">
                        <h4>RESTART / INITIALIZE</h4>
                        <p>AFTER DEPLOYING BINARY SCHEMAS, SELECT THE ARTIFACTS ABOVE TO ONLINE THE DYNAMIC MONITOR ENGINE.</p>
                      </div>
                    </div>
                  </div>
                </section>
              </div>
            )}
          </main>
        </div>
      )}
    </div>
  );
}

export default App;
