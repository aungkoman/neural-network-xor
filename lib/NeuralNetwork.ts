
import { SerializedNetwork } from '../types';

class Matrix {
  rows: number;
  cols: number;
  data: number[][];

  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(this.rows).fill(0).map(() => Array(this.cols).fill(0));
  }

  static fromArray(arr: number[]): Matrix {
    let m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      m.data[i][0] = arr[i];
    }
    return m;
  }
  
  toArray(): number[] {
    let arr: number[] = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  randomize(): this {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = Math.random() * 2 - 1;
      }
    }
    return this;
  }

  add(n: Matrix | number): this {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.cols !== n.cols) {
        throw new Error('Matrix dimensions must match for addition.');
      }
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += n;
        }
      }
    }
    return this;
  }

  static subtract(a: Matrix, b: Matrix): Matrix {
    if (a.rows !== b.rows || a.cols !== b.cols) {
        throw new Error('Matrix dimensions must match for subtraction.');
    }
    let result = new Matrix(a.rows, a.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        result.data[i][j] = a.data[i][j] - b.data[i][j];
      }
    }
    return result;
  }

  static transpose(matrix: Matrix): Matrix {
    let result = new Matrix(matrix.cols, matrix.rows);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[j][i] = matrix.data[i][j];
      }
    }
    return result;
  }

  static multiply(a: Matrix, b: Matrix): Matrix {
    if (a.cols !== b.rows) {
      throw new Error('Columns of A must match rows of B for dot product.');
    }
    let result = new Matrix(a.rows, b.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }
  
  multiply(n: Matrix | number): this {
    if (n instanceof Matrix) { // Element-wise multiplication
         if (this.rows !== n.rows || this.cols !== n.cols) {
            throw new Error('Matrix dimensions must match for element-wise multiplication.');
         }
         for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] *= n.data[i][j];
            }
         }
    } else { // Scalar multiplication
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] *= n;
            }
        }
    }
    return this;
  }

  map(fn: (x: number) => number): this {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = fn(this.data[i][j]);
      }
    }
    return this;
  }

  static map(matrix: Matrix, fn: (x: number) => number): Matrix {
    let result = new Matrix(matrix.rows, matrix.cols);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[i][j] = fn(matrix.data[i][j]);
      }
    }
    return result;
  }
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y: number): number {
  return y * (1 - y);
}

export class NeuralNetwork {
  inputNodes: number;
  hiddenNodes: number;
  outputNodes: number;
  
  weights_ih: Matrix;
  weights_ho: Matrix;
  bias_h: Matrix;
  bias_o: Matrix;
  learningRate: number;

  constructor(input: number, hidden: number, output: number) {
    this.inputNodes = input;
    this.hiddenNodes = hidden;
    this.outputNodes = output;
    
    this.weights_ih = new Matrix(this.hiddenNodes, this.inputNodes).randomize();
    this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes).randomize();
    
    this.bias_h = new Matrix(this.hiddenNodes, 1).randomize();
    this.bias_o = new Matrix(this.outputNodes, 1).randomize();
    this.learningRate = 0.1;
  }

  setLearningRate(rate: number): void {
      this.learningRate = rate;
  }

  predict(input_array: number[]): number[] {
    const inputs = Matrix.fromArray(input_array);
    const hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(sigmoid);

    const output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(sigmoid);

    return output.toArray();
  }

  train(input_array: number[], target_array: number[]): void {
    const inputs = Matrix.fromArray(input_array);
    const targets = Matrix.fromArray(target_array);

    // Feedforward
    const hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(sigmoid);

    const outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(sigmoid);

    // Backpropagation
    // Calculate output errors
    const output_errors = Matrix.subtract(targets, outputs);

    // Calculate output gradient
    const gradients = Matrix.map(outputs, dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learningRate);

    // Calculate deltas for hidden-to-output weights
    const hidden_t = Matrix.transpose(hidden);
    const weight_ho_deltas = Matrix.multiply(gradients, hidden_t);

    // Adjust the weights and biases
    this.weights_ho.add(weight_ho_deltas);
    this.bias_o.add(gradients);

    // Calculate hidden layer errors
    const who_t = Matrix.transpose(this.weights_ho);
    const hidden_errors = Matrix.multiply(who_t, output_errors);

    // Calculate hidden gradient
    const hidden_gradient = Matrix.map(hidden, dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learningRate);

    // Calculate deltas for input-to-hidden weights
    const inputs_t = Matrix.transpose(inputs);
    const weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_t);

    // Adjust the weights and biases
    this.weights_ih.add(weight_ih_deltas);
    this.bias_h.add(hidden_gradient);
  }

  serialize(): SerializedNetwork {
    return {
      inputNodes: this.inputNodes,
      hiddenNodes: this.hiddenNodes,
      outputNodes: this.outputNodes,
      weights_ih: this.weights_ih.data,
      weights_ho: this.weights_ho.data,
      bias_h: this.bias_h.data,
      bias_o: this.bias_o.data,
      learningRate: this.learningRate,
    };
  }

  deserialize(data: SerializedNetwork): void {
    this.inputNodes = data.inputNodes;
    this.hiddenNodes = data.hiddenNodes;
    this.outputNodes = data.outputNodes;
    this.learningRate = data.learningRate;
    
    this.weights_ih = new Matrix(this.hiddenNodes, this.inputNodes);
    this.weights_ih.data = data.weights_ih;

    this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes);
    this.weights_ho.data = data.weights_ho;
    
    this.bias_h = new Matrix(this.hiddenNodes, 1);
    this.bias_h.data = data.bias_h;
    
    this.bias_o = new Matrix(this.outputNodes, 1);
    this.bias_o.data = data.bias_o;
  }
}
