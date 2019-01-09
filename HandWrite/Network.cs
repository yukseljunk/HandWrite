using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

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

    }
}