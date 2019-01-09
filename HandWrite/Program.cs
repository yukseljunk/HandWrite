using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace HandWrite
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new Network(new List<int>() { 28 * 28, 16, 16, 10 });
            FillWeightsAndBalancesRandomly(network);

            //create a nice 1
            var oneInput = new DenseMatrix(28, 28);
            for (int i = 6; i < 22; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    oneInput[i, 13 + j] = 0.95 + (0.01 * (j % 2));
                }
            }

            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    Console.Write(oneInput[i, j]);
                    Console.Write(new string(' ', 4 - oneInput[i, j].ToString().Length));
                }
                Console.WriteLine("");
            }

            Train(network, oneInput,1, ErrFunction);
            Console.Read();
        }

        private static double ErrFunction(Matrix<double> outputMatrix, int expectedValue)
        {
            double result = 0;
            for (int i = 0; i < outputMatrix.RowCount; i++)
            {
                var rowValue = outputMatrix[i, 0];
                if(i==expectedValue-1)
                {
                    result += Math.Sqrt(1 - rowValue);
                }
                else
                {
                    result += Math.Sqrt(rowValue);
                }
            }
            return result;
        }

        static void FillWeightsAndBalancesRandomly(Network network)
        {
            Random random = new Random();

            foreach (var weightMatrix in network.WeightMatrices)
            {
                for (int i = 0; i < weightMatrix.RowCount; i++)
                {
                    for (int j = 0; j < weightMatrix.ColumnCount; j++)
                    {
                        weightMatrix[i, j] = random.NextDouble(-9.99, 9.99);
                    }
                }
            }
            foreach (var layer in network.Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuron.Bias = random.NextDouble(-9.99, 9.99);
                }
            }

        }

        static double Train(Network network, Matrix<double> inputActivationValues, int expectedResult, Func<Matrix<double>, int, double> errorFunction)
        {
            network.Layers.First.Value.ActivationMatrix = inputActivationValues;
            network.FeedForward();
            return errorFunction(network.Layers.Last.Value.ActivationMatrix, expectedResult);
        }

    }
}
