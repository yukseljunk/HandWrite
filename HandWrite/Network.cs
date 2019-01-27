using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace HandWrite
{
    public class Network
    {
        /// <summary>
        /// Weight matrices between layers
        /// First element is the weight matrix between first layer and second layer and so on 
        /// </summary>
        public List<Matrix<double>> WeightMatrices = new List<Matrix<double>>();

        /// <summary>
        /// Layers in network, first is input, last is output
        /// </summary>
        public LinkedList<Layer> Layers = new LinkedList<Layer>();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="layerSizes"> Node counts in layers, size of this gives the number of layers</param>
        public Network(List<int> layerSizes)
        {
            LayerSizes = layerSizes;
            foreach (int size in layerSizes)
            {
                Layers.AddLast(new Layer(size));
            }
            for (int i = 1; i < layerSizes.Count; i++)
            {
                WeightMatrices.Add(Matrix<double>.Build.Dense(layerSizes[i], layerSizes[i - 1]));
            }
        }

        public List<int> LayerSizes { get; set; }

        /// <summary>
        /// Next activation matrix is sigmoid(weight matrix * current activation matrix + next bias matrix)
        /// assumes that all the weight and bias matrices have values, random most probably plus the first activation
        /// </summary>
        public void FeedForward()
        {
            var weightMatrixIndex = 0;
            var refNode = Layers.First;
            while (refNode != null && refNode.Next != null)
            {
                refNode.Next.Value.ActivationMatrix =
                    WeightMatrices[weightMatrixIndex].Multiply(refNode.Value.ActivationMatrix) +
                    refNode.Next.Value.BiasMatrix;
                refNode.Next.Value.Reluize();
                refNode = refNode.Next;
                weightMatrixIndex++;
            }
        }

        public override bool Equals(object obj)
        {
            var other = (Network)obj;
            if (this.WeightMatrices.Count != other.WeightMatrices.Count) return false;
            for (int i = 0; i < WeightMatrices.Count; i++)
            {
                if (WeightMatrices[i].Equals(other.WeightMatrices[i]))
                {
                    return false;
                }
            }
            if (this.Layers.Count != other.Layers.Count) return false;
            for (int i = 0; i < Layers.Count; i++)
            {
                if (!Layers.ToList()[i].Equals(other.Layers.ToList()[i]))
                {
                    return false;
                }
            }
            return true;
        }
        public override int GetHashCode()
        {
            int layerAvg = 1;
            if (Layers.Any())
            {
                layerAvg = Layers.ToList().Sum(l => l.GetHashCode()) / Layers.Count;
            }

            int weightSumAvg = 1;
            if (WeightMatrices.Any())
            {
                weightSumAvg = WeightMatrices.Sum(w => w.GetHashCode()) / WeightMatrices.Count;
            }
            return layerAvg * weightSumAvg;
        }

        public void DeSerialize(string filename)
        {
            if (!File.Exists(filename)) throw new FileNotFoundException("Deserialize file not found", filename);
            WeightMatrices = new List<Matrix<double>>();
            var layers = new LinkedList<Layer>();
            var allLines = File.ReadAllLines(filename);
            int rows = 0;
            int cols = 0;
            var weightMatrixActive = true;
            var firstLayerSize = 0;
            var activeRowNo = 0;
            for (int i = 0; i < allLines.Length; i++)
            {
                var line = allLines[i];
                if (line.Trim() == "") break;
                if (line.StartsWith("w") || line.StartsWith("b"))
                {
                    activeRowNo = 0;
                    var matrixRc = line.Replace("w", "").Replace("b", "").Split(new char[] { 'x' }, StringSplitOptions.RemoveEmptyEntries);
                    rows = int.Parse(matrixRc[0]);
                    cols = int.Parse(matrixRc[1]);
                    weightMatrixActive = line.StartsWith("w");
                    if (weightMatrixActive)
                    {
                        WeightMatrices.Add(new DenseMatrix(rows, cols));
                        if (firstLayerSize == 0)
                        {
                            firstLayerSize = cols;
                            layers.AddFirst(new Layer(firstLayerSize));
                            LayerSizes.Add(firstLayerSize);
                        }
                    }
                    else
                    {
                        layers.AddLast(new Layer(rows));
                        LayerSizes.Add(rows);
                    }
                    continue;
                }
                if (weightMatrixActive)
                {
                    var weightMatrix = WeightMatrices.Last();
                    var vals = line.Split(new char[] { ' ' });
                    for (int j = 0; j < weightMatrix.ColumnCount; j++)
                    {
                        weightMatrix[activeRowNo, j] = double.Parse(vals[j]);
                    }
                }
                else//bias matrix
                {
                    var vals = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    layers.Last.Value.Neurons[activeRowNo].Bias = double.Parse(vals[0]);
                }
                activeRowNo++;
            }
            Layers = layers;
        }

        public void Serialize(string filename)
        {
            if (File.Exists(filename)) File.Delete(filename);

            foreach (var weightMatrix in WeightMatrices)
            {
                File.AppendAllText(filename, "w" + weightMatrix.RowCount.ToString() + "x" + weightMatrix.ColumnCount.ToString());
                File.AppendAllText(filename, Environment.NewLine);
                foreach (var row in weightMatrix.EnumerateRows())
                {
                    for (int i = 0; i < weightMatrix.ColumnCount; i++)
                    {
                        File.AppendAllText(filename, row[i].ToString() + " ");
                    }
                    File.AppendAllText(filename, Environment.NewLine);
                }
            }
            var layerNo = 0;
            foreach (var layer in Layers)
            {
                layerNo++;
                if (layerNo == 1) continue;
                File.AppendAllText(filename, "b" + layer.BiasMatrix.RowCount.ToString() + "x" + layer.BiasMatrix.ColumnCount.ToString());
                File.AppendAllText(filename, Environment.NewLine);
                foreach (var row in layer.BiasMatrix.EnumerateRows())
                {
                    for (int i = 0; i < layer.BiasMatrix.ColumnCount; i++)
                    {
                        File.AppendAllText(filename, row[i].ToString() + " ");
                    }
                    File.AppendAllText(filename, Environment.NewLine);
                }
            }
        }

    }
}