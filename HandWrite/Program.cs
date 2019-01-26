using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace HandWrite
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new Network(new List<int>() { 28 * 28, 16, 16, 10 });
            FillWeightsAndBalancesRandomly(network);

            /*network.Serialize(@"c:\temp\serializedmatrix.txt");
            network.DeSerialize(@"c:\temp\serializedmatrix.txt");
            */

            var fileTemplate = @"C:\temp\hwinput\input{0}.txt";
            bool writeInputToFile = false;
            var dataFactory = new DataFactory();
            var imageIndexor = 0;
            double sumErrorRate = 0.0;
            foreach (var denseMatrix in dataFactory.TrainingData())
            {
                var file = string.Format(fileTemplate, imageIndexor);
                imageIndexor++;
                var image = denseMatrix.Item1;
                var label = denseMatrix.Item2;

                //Console.WriteLine(label);
                if (writeInputToFile)
                {
                    for (int i = 0; i < 28; i++)
                    {
                        for (int j = 0; j < 28; j++)
                        {
                            File.AppendAllText(file, image[i, j].ToString());
                            File.AppendAllText(file, new string(' ', 6 - image[i, j].ToString().Length));
                        }
                        File.AppendAllText(file, Environment.NewLine);
                    }
                }
                var errorRate = Train(network, image, label, ErrFunction);
                sumErrorRate += errorRate;
                Console.WriteLine("Img Index: {2},Err: {0}, AvgErr: {1}", errorRate, sumErrorRate / (double)imageIndexor, imageIndexor);

            }
            Console.ReadKey();

        }

        private static double ErrFunction(Matrix<double> outputMatrix, int expectedValue)
        {
            double result = 0;
            for (int i = 0; i < outputMatrix.RowCount; i++)
            {
                var rowValue = outputMatrix[i, 0];
                if (i == expectedValue - 1)
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
            var layerNo = 0;
            foreach (var layer in network.Layers)
            {
                layerNo++;
                if (layerNo == 1) continue;
                foreach (var neuron in layer.Neurons)
                {
                    neuron.Bias = random.NextDouble(-9.99, 9.99);
                }
            }

        }

        static double Train(Network network, MathNet.Numerics.LinearAlgebra.Matrix<double> inputActivationValues, int expectedResult, Func<Matrix<double>, int, double> errorFunction)
        {
            network.Layers.First.Value.ActivationMatrix = inputActivationValues;
            network.FeedForward();
            return errorFunction(network.Layers.Last.Value.ActivationMatrix, expectedResult);
        }

    }
}
