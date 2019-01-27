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

            network.Serialize(@"c:\temp\serializedmatrix.txt");

            //var network2 = new Network(new List<int>());
            //network2.DeSerialize(@"c:\temp\serializedmatrix.txt");
            //Console.WriteLine(network.Equals(network2));

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
                Console.WriteLine("Img Index: {2}, Label: {3},Err: {0}, AvgErr: {1}", errorRate, sumErrorRate / (double)imageIndexor, imageIndexor, label);
                //if (imageIndexor == 1000) break;

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
                        weightMatrix[i, j] = Math.Round(RandomGaussian(random) * 10, 4);
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
                    neuron.Bias = Math.Round(RandomGaussian(random) * 10, 4);
                }
            }

        }

        static double RandomGaussian(Random rand, double mean = 0.0, double stdDev = 1.0)
        {
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
        }

        static double Train(Network network, MathNet.Numerics.LinearAlgebra.Matrix<double> inputActivationValues, int expectedResult, Func<Matrix<double>, int, double> errorFunction)
        {
            network.Layers.First.Value.ActivationMatrix = inputActivationValues;
            network.FeedForward();
            return errorFunction(network.Layers.Last.Value.ActivationMatrix, expectedResult);
        }

    }
}
