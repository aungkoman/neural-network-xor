import React, { useState, useRef, useEffect, useCallback } from 'react';
import { NeuralNetwork } from './lib/NeuralNetwork';
import NeuralNetworkVisualizer from './components/NeuralNetworkVisualizer';
import { SerializedNetwork } from './types';
import { 
  nameDataset, 
  nameToVector, 
  genderToVector, 
  vectorToPrediction,
  VOCAB_SIZE,
  GENDERS
} from './lib/DataHelper';


// Configuration
const HIDDEN_NODES = 15;
const OUTPUT_NODES = GENDERS.length; // 'male', 'female'

const App: React.FC = () => {
  const [learningRate, setLearningRate] = useState(0.05);
  const [isRunning, setIsRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [error, setError] = useState(0);
  const [networkData, setNetworkData] = useState<SerializedNetwork | null>(null);
  const [predictions, setPredictions] = useState<( { gender: string, confidence: number } | null)[]>([]);
  const [testName, setTestName] = useState('အောင်ကျော်');
  const [testPrediction, setTestPrediction] = useState<{ gender: string, confidence: number } | null>(null);


  const networkRef = useRef<NeuralNetwork>(new NeuralNetwork(VOCAB_SIZE, HIDDEN_NODES, OUTPUT_NODES));
  const animationFrameId = useRef<number>();

  const updateTestPrediction = useCallback((name: string) => {
      if (!networkRef.current) return;
      const inputVector = nameToVector(name);
      const outputVector = networkRef.current.predict(inputVector);
      setTestPrediction(vectorToPrediction(outputVector));
  }, []);

  const updatePredictions = useCallback(() => {
    if (!networkRef.current) return;
    const newPredictions = nameDataset.map(d => {
      const inputVector = nameToVector(d.name);
      const outputVector = networkRef.current.predict(inputVector);
      return vectorToPrediction(outputVector);
    });
    setPredictions(newPredictions);
  }, []);

  const initializeNetwork = useCallback(() => {
    networkRef.current = new NeuralNetwork(VOCAB_SIZE, HIDDEN_NODES, OUTPUT_NODES);
    networkRef.current.setLearningRate(learningRate);
    setNetworkData(networkRef.current.serialize());
    setEpoch(0);
    setError(0);
    updatePredictions();
    updateTestPrediction(testName);
  }, [learningRate, testName, updatePredictions, updateTestPrediction]);

  useEffect(() => {
    initializeNetwork();
  }, [initializeNetwork]);

  const trainLoop = useCallback(() => {
    if (!networkRef.current) return;

    // Train on a batch of 10 for smoother learning per frame
    let batchError = 0;
    for (let i = 0; i < 10; i++) {
        const data = nameDataset[Math.floor(Math.random() * nameDataset.length)];
        const inputs = nameToVector(data.name);
        const targets = genderToVector(data.gender);
        networkRef.current.train(inputs, targets);
        
        // Calculate error for this single training example
        const prediction = networkRef.current.predict(inputs);
        const err1 = targets[0] - prediction[0];
        const err2 = targets[1] - prediction[1];
        batchError += (err1 * err1 + err2 * err2) / 2;
    }
    
    setEpoch(prev => prev + 10);
    setError(batchError / 10);


    if (epoch % 100 === 0) {
      updatePredictions();
      updateTestPrediction(testName);
      setNetworkData(networkRef.current.serialize());
    }

    animationFrameId.current = requestAnimationFrame(trainLoop);
  }, [epoch, testName, updatePredictions, updateTestPrediction]);

  useEffect(() => {
    if (isRunning) {
      animationFrameId.current = requestAnimationFrame(trainLoop);
    } else {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    }
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [isRunning, trainLoop]);

  const handleStartStop = () => {
    setIsRunning(!isRunning);
    if (isRunning) { // If it was running, update everything on stop
        updatePredictions();
        updateTestPrediction(testName);
        setNetworkData(networkRef.current.serialize());
    }
  };

  const handleLearningRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newRate = parseFloat(e.target.value);
    setLearningRate(newRate);
    networkRef.current.setLearningRate(newRate);
  };
  
  const handleSaveModel = () => {
    const serialized = networkRef.current.serialize();
    localStorage.setItem('myanmarNameClassifierModel', JSON.stringify(serialized));
    alert('Model saved!');
  };

  const handleLoadModel = () => {
    const savedModel = localStorage.getItem('myanmarNameClassifierModel');
    if (savedModel) {
      const serialized = JSON.parse(savedModel);
      networkRef.current.deserialize(serialized);
      setNetworkData(networkRef.current.serialize());
      setLearningRate(serialized.learningRate);
      updatePredictions();
      updateTestPrediction(testName);
      alert('Model loaded!');
    } else {
      alert('No saved model found.');
    }
  };
  
  const handleTestNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      setTestName(e.target.value);
      updateTestPrediction(e.target.value);
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col items-center p-4 lg:p-8">
      <header className="w-full text-center mb-6">
        <h1 className="text-4xl lg:text-5xl font-bold text-cyan-400">Myanmar Name Gender Classifier</h1>
        <p className="text-gray-400 mt-2">A neural network learning to predict gender from a name.</p>
      </header>

      <main className="w-full max-w-7xl flex flex-col lg:flex-row gap-8">
        <div className="flex-grow bg-gray-800 border border-cyan-500/30 rounded-lg shadow-lg p-4 flex flex-col items-center justify-center min-h-[500px]">
          {networkData && <NeuralNetworkVisualizer network={networkData} />}
        </div>

        <aside className="w-full lg:w-96 flex-shrink-0 space-y-6">
          
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-cyan-500/30">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Live Test</h2>
            <div className="space-y-4">
              <div>
                <label htmlFor="testName" className="block mb-2 font-semibold text-gray-400">Enter a Name:</label>
                <input
                  type="text"
                  id="testName"
                  value={testName}
                  onChange={handleTestNameChange}
                  className="w-full bg-gray-700 text-white p-2 rounded-lg border-2 border-gray-600 focus:border-cyan-500 focus:outline-none"
                  placeholder="e.g., စုစု"
                />
              </div>
              {testPrediction && (
                <div className="text-center p-4 bg-gray-700/50 rounded-lg">
                  <div className="text-lg text-gray-400">Prediction</div>
                  <div className={`text-3xl font-bold ${testPrediction.gender === 'male' ? 'text-blue-400' : 'text-pink-400'}`}>
                    {testPrediction.gender.charAt(0).toUpperCase() + testPrediction.gender.slice(1)}
                  </div>
                  <div className="text-md text-gray-500">
                    ({(testPrediction.confidence * 100).toFixed(1)}% confidence)
                  </div>
                </div>
              )}
            </div>
          </div>
        
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-cyan-500/30">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Controls</h2>
            <div className="space-y-4">
              <button
                onClick={handleStartStop}
                className={`w-full py-3 px-4 rounded-lg font-bold text-lg transition-all duration-200 ${isRunning ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'}`}>
                {isRunning ? 'Stop Training' : 'Start Training'}
              </button>
              <button
                onClick={initializeNetwork}
                className="w-full py-3 px-4 rounded-lg font-bold text-lg bg-blue-500 hover:bg-blue-600 transition-all duration-200">
                Reset Network
              </button>
              <div className="flex gap-2">
                <button
                    onClick={handleSaveModel}
                    className="w-full py-2 px-4 rounded-lg font-semibold bg-gray-600 hover:bg-gray-700 transition-all duration-200">
                    Save Model
                </button>
                <button
                    onClick={handleLoadModel}
                    className="w-full py-2 px-4 rounded-lg font-semibold bg-gray-600 hover:bg-gray-700 transition-all duration-200">
                    Load Model
                </button>
              </div>
              <div>
                <label htmlFor="learningRate" className="block mb-2 font-semibold text-gray-400">Learning Rate: {learningRate.toFixed(3)}</label>
                <input
                  type="range"
                  id="learningRate"
                  min="0.001"
                  max="0.5"
                  step="0.001"
                  value={learningRate}
                  onChange={handleLearningRateChange}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-cyan-500/30">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Statistics</h2>
            <div className="space-y-2 text-lg">
              <div className="flex justify-between">
                <span className="font-semibold text-gray-400">Epochs:</span>
                <span className="font-mono">{epoch.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="font-semibold text-gray-400">Batch Error:</span>
                <span className="font-mono">{error.toFixed(6)}</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-cyan-500/30">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Training Performance</h2>
            <table className="w-full text-left font-mono text-sm">
              <thead>
                <tr className="border-b border-gray-600">
                  <th className="p-2">Name</th>
                  <th className="p-2">Target</th>
                  <th className="p-2">Output</th>
                </tr>
              </thead>
              <tbody>
                {nameDataset.map((data, i) => {
                  const prediction = predictions[i];
                  if (!prediction) return null;
                  const isCorrect = prediction.gender === data.gender;
                  
                  return (
                    <tr key={i} className="border-b border-gray-700 last:border-0">
                      <td className="p-2 text-white">{data.name}</td>
                      <td className={`p-2 font-bold ${data.gender === 'male' ? 'text-blue-400' : 'text-pink-400'}`}>{data.gender}</td>
                      <td className={`p-2 font-bold ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>
                        {prediction.gender} ({(prediction.confidence * 100).toFixed(0)}%)
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </aside>
      </main>
    </div>
  );
};

export default App;
