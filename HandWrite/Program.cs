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
            var network = new Network(new List<int>() { 1, 1, 1 });
            network.WeightMatrices[0][0, 0] = 3;
            network.WeightMatrices[1][0, 0] = 2;

            var layerNo = 0;
            foreach (var layer in network.Layers)
            {
                layerNo++;
                if (layerNo == 1) continue;
                var bias = 0;
                if (layerNo == 2) bias = 1;
                layer.Neurons[0].Bias = bias;
            }
            Random random = new Random();
            //f(x)=x;
            var count = 100;
            for (int i = 1; i < count; i++)
            {
                var y = (double)i / count;
                network.Layers.First.Value.Neurons[0].Activation = y;

                network.FeedForward();

                //back-propagate
                var alm2 = network.Layers.Last.Previous.Previous.Value.Neurons[0].Activation;
                var alm1 = network.Layers.Last.Previous.Value.Neurons[0].Activation;
                var al = network.Layers.Last.Value.Neurons[0].Activation;
                var bl = network.Layers.Last.Value.Neurons[0].Bias;
                var blm1 = network.Layers.Last.Previous.Value.Neurons[0].Bias;
                var wl = network.WeightMatrices[1][0, 0];
                var wlm1 = network.WeightMatrices[0][0, 0];

                var zl = alm1 * wl + bl;
                var zlm1 = alm2 * wlm1 + blm1;

                if (y == al)
                {

                    var x1 = 1;
                    var y1 = x1;
                }
                var cwl = 2 * (y - al) * alm1 * SigmoidDiff(zl);
                var cbl = 2 * (y - al) * SigmoidDiff(zl);
                var cwlm1 = 2 * (y - al) * wl * alm2 * SigmoidDiff(zl) * SigmoidDiff(zl - 1);
                var cblm1 = 2 * (y - al) * wl * SigmoidDiff(zl) * SigmoidDiff(zl - 1);

                var k = Math.Abs(RandomGaussian(random, 0.05, 0.05))/i;
                //var k = 0.0003/(double)i;

                network.Layers.Last.Value.Neurons[0].Bias = bl - k * cbl * Math.Sign(cbl)*-1;
                network.WeightMatrices[1][0, 0] = wl - k * cwl * Math.Sign(cwl) * -1;
                network.Layers.Last.Previous.Value.Neurons[0].Bias = blm1 - k * cblm1 * Math.Sign(cblm1) * -1;
                network.WeightMatrices[0][0, 0] = wlm1 - k * cwlm1 * Math.Sign(cwlm1) * -1;

                Console.WriteLine("Iter {0} Error {1} delta {2}", i, (y - al).ToString(), k);
            }

            Console.ReadLine();
        }

        private static double SigmoidDiff(double val)
        {
            return Layer.Sigmoid(val) * (1 - Layer.Sigmoid(val));
        }

        static void MainOld(string[] args)
        {

            var network = new Network(new List<int>() { 28 * 28, 16, 16, 10 });
            FillWeightsAndBalancesRandomly(network);

            var miniBatchSize = 10;
            var epochCount = 3;

            // network.Serialize(@"c:\temp\serializedmatrix.txt");

            //var network2 = new Network(new List<int>());
            //network2.DeSerialize(@"c:\temp\serializedmatrix.txt");
            //Console.WriteLine(network.Equals(network2));

            var fileTemplate = @"C:\temp\hwinput\input{0}.txt";
            bool writeInputToFile = false;

            for (int k = 1; k <= epochCount; k++)
            {
                Console.WriteLine("Epoch " + k);

                var dataFactory = new DataFactory();
                double sumErrorRate = 0.0;
                var imageIndexor = 0;

                foreach (var batch in dataFactory.TrainingData(miniBatchSize))
                {
                    foreach (var denseMatrix in batch)
                    {
                        var image = denseMatrix.Item1;
                        var label = denseMatrix.Item2;
                        imageIndexor++;
                        var file = string.Format(fileTemplate, imageIndexor);
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
                        Console.WriteLine("Img Index: {2}, Label: {3},Err: {0}, AvgErr: {1}", errorRate,
                                          sumErrorRate / (double)imageIndexor, imageIndexor, label);

                        if (imageIndexor == 100) break;
                    }
                    if (imageIndexor == 100) break;

                }
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
