
export interface TrainingData {
  inputs: number[];
  targets: number[];
}

export interface SerializedNetwork {
  inputNodes: number;
  hiddenNodes: number;
  outputNodes: number;
  weights_ih: number[][];
  weights_ho: number[][];
  bias_h: number[][];
  bias_o: number[][];
  learningRate: number;
}
