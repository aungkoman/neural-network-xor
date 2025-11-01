
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { NeuralNetwork } from './lib/NeuralNetwork';
import NeuralNetworkVisualizer from './components/NeuralNetworkVisualizer';
import { TrainingData, SerializedNetwork } from './types';

const xorData: TrainingData[] = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [1] },
  { inputs: [1, 0], targets: [1] },
  { inputs: [1, 1], targets: [0] },
];

const App: React.FC = () => {
  const [learningRate, setLearningRate] = useState(0.1);
  const [isRunning, setIsRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [error, setError] = useState(0);
  const [networkData, setNetworkData] = useState<SerializedNetwork | null>(null);
  const [predictions, setPredictions] = useState<number[][]>([]);

  const networkRef = useRef<NeuralNetwork>(new NeuralNetwork(2, 4, 1));
  const animationFrameId = useRef<number>();

  const initializeNetwork = useCallback(() => {
    networkRef.current = new NeuralNetwork(2, 4, 1);
    networkRef.current.setLearningRate(learningRate);
    setNetworkData(networkRef.current.serialize());
    setEpoch(0);
    setError(0);
    updatePredictions();
  }, [learningRate]);

  useEffect(() => {
    initializeNetwork();
  }, [initializeNetwork]);

  const updatePredictions = () => {
    const newPredictions = xorData.map(d => networkRef.current.predict(d.inputs));
    setPredictions(newPredictions);
  };

  const trainLoop = useCallback(() => {
    if (!networkRef.current) return;

    for (let i = 0; i < 10; i++) { // 10 iterations per frame for speed
        const data = xorData[Math.floor(Math.random() * xorData.length)];
        networkRef.current.train(data.inputs, data.targets);
    }
    
    setEpoch(prev => prev + 10);

    if (epoch % 100 === 0) {
      let currentError = 0;
      for (const data of xorData) {
        const prediction = networkRef.current.predict(data.inputs);
        const err = data.targets[0] - prediction[0];
        currentError += err * err;
      }
      setError(currentError / xorData.length);
      updatePredictions();
      setNetworkData(networkRef.current.serialize());
    }

    animationFrameId.current = requestAnimationFrame(trainLoop);
  }, [epoch]);

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
     // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRunning, trainLoop]);

  const handleStartStop = () => {
    setIsRunning(!isRunning);
    if (!isRunning) {
        setNetworkData(networkRef.current.serialize());
        updatePredictions();
    }
  };

  const handleLearningRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newRate = parseFloat(e.target.value);
    setLearningRate(newRate);
    networkRef.current.setLearningRate(newRate);
  };
  
  const handleSaveModel = () => {
    const serialized = networkRef.current.serialize();
    localStorage.setItem('neuralNetworkModel', JSON.stringify(serialized));
    alert('Model saved!');
  };

  const handleLoadModel = () => {
    const savedModel = localStorage.getItem('neuralNetworkModel');
    if (savedModel) {
      const serialized = JSON.parse(savedModel);
      networkRef.current.deserialize(serialized);
      setNetworkData(networkRef.current.serialize());
      setLearningRate(serialized.learningRate);
      updatePredictions();
      alert('Model loaded!');
    } else {
      alert('No saved model found.');
    }
  };


  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col items-center p-4 lg:p-8">
      <header className="w-full text-center mb-6">
        <h1 className="text-4xl lg:text-5xl font-bold text-cyan-400">Neural Network XOR Visualizer</h1>
        <p className="text-gray-400 mt-2">Training a simple neural network from scratch with backpropagation.</p>
      </header>

      <main className="w-full max-w-7xl flex flex-col lg:flex-row gap-8">
        <div className="flex-grow bg-gray-800 border border-cyan-500/30 rounded-lg shadow-lg p-4 flex flex-col items-center justify-center">
          {networkData && <NeuralNetworkVisualizer network={networkData} />}
        </div>

        <aside className="w-full lg:w-96 flex-shrink-0 space-y-6">
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
                <span className="font-semibold text-gray-400">Error:</span>
                <span className="font-mono">{error.toFixed(6)}</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-cyan-500/30">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Predictions</h2>
            <table className="w-full text-left font-mono">
              <thead>
                <tr className="border-b border-gray-600">
                  <th className="p-2">Input</th>
                  <th className="p-2">Target</th>
                  <th className="p-2">Output</th>
                </tr>
              </thead>
              <tbody>
                {xorData.map((data, i) => {
                  const output = predictions[i] ? predictions[i][0] : 0;
                  const bgColor = `rgba(34, 211, 238, ${Math.abs(data.targets[0] - output)})`;
                  return (
                    <tr key={i} className="border-b border-gray-700 last:border-0" style={{ backgroundColor: bgColor }}>
                      <td className="p-2 text-white">[{data.inputs.join(', ')}]</td>
                      <td className="p-2 text-white">{data.targets[0]}</td>
                      <td className={`p-2 font-bold ${Math.round(output) === data.targets[0] ? 'text-green-400' : 'text-red-400'}`}>{output.toFixed(4)}</td>
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
