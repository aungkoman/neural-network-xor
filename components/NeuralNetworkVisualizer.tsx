
import React, { useRef, useEffect } from 'react';
import { SerializedNetwork } from '../types';

interface NeuralNetworkVisualizerProps {
  network: SerializedNetwork;
}

const NeuralNetworkVisualizer: React.FC<NeuralNetworkVisualizerProps> = ({ network }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const parent = canvas.parentElement;
    if(!parent) return;

    const width = parent.clientWidth;
    const height = 500; // Fixed height
    canvas.width = width;
    canvas.height = height;

    ctx.clearRect(0, 0, width, height);

    const neuronRadius = 12;
    const layerMargin = 150;
    const layerWidth = width - 2 * layerMargin;
    const layerXPositions = [layerMargin, layerMargin + layerWidth / 2, layerMargin + layerWidth];

    const layersNodeCounts = [network.inputNodes, network.hiddenNodes, network.outputNodes];
    const layerPositions: { x: number; y: number }[][] = [];

    // Calculate neuron positions
    layersNodeCounts.forEach((nodeCount, i) => {
      const layerX = layerXPositions[i];
      const layerNodes: { x: number; y: number }[] = [];
      const verticalSpacing = height / (nodeCount + 1);
      for (let j = 0; j < nodeCount; j++) {
        const neuronY = verticalSpacing * (j + 1);
        layerNodes.push({ x: layerX, y: neuronY });
      }
      layerPositions.push(layerNodes);
    });

    // Draw connections (weights)
    const drawConnections = (fromLayer: {x:number, y:number}[], toLayer: {x:number, y:number}[], weights: number[][]) => {
      fromLayer.forEach((fromNode, i) => {
        toLayer.forEach((toNode, j) => {
          const weight = weights[j][i];
          const alpha = Math.min(1, Math.abs(weight) * 2);

          ctx.beginPath();
          ctx.moveTo(fromNode.x, fromNode.y);
          ctx.lineTo(toNode.x, toNode.y);
          
          // Green for positive, red for negative
          ctx.strokeStyle = weight > 0 ? `rgba(16, 185, 129, ${alpha})` : `rgba(239, 68, 68, ${alpha})`;
          ctx.lineWidth = Math.min(8, Math.abs(weight) * 5);
          ctx.stroke();
        });
      });
    };
    
    if(layerPositions.length >= 3) {
      drawConnections(layerPositions[0], layerPositions[1], network.weights_ih);
      drawConnections(layerPositions[1], layerPositions[2], network.weights_ho);
    }

    // Draw neurons
    layerPositions.forEach((layer) => {
      layer.forEach((node) => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, neuronRadius, 0, 2 * Math.PI);
        ctx.fillStyle = '#1f2937'; // bg-gray-800
        ctx.fill();
        ctx.strokeStyle = '#06b6d4'; // cyan-500
        ctx.lineWidth = 3;
        ctx.stroke();
      });
    });

  }, [network]);

  return <canvas ref={canvasRef} />;
};

export default NeuralNetworkVisualizer;
