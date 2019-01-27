using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace HandWrite
{
    public class Layer
    {
        public Layer(int neuronAmount)
        {
            Neurons = new List<Neuron>(neuronAmount);
            for (int i = 0; i < neuronAmount; i++)
            {
                Neurons.Add(new Neuron());
            }
        }
        public List<Neuron> Neurons { get; set; }

        public Neuron HighestActivated
        {
            get
            {
                return Neurons.FirstOrDefault(n => n.Activation == Neurons.Max(ne => ne.Activation));
            }
        }

        public Matrix<double> ActivationMatrix
        {
            get
            {
                if (Neurons == null) return null;
                if (!Neurons.Any()) return null;

                return Matrix<double>.Build.Dense(Neurons.Count, 1, Neurons.Select(n => n.Activation).ToArray());
            }
            set
            {
                var nIndex = 0;
                foreach (var activation in value.Storage.Enumerate())
                {
                    Neurons[nIndex].Activation = activation;
                    nIndex++;
                }
            }
        }

        public override bool Equals(object obj)
        {
            var other = (Layer)obj;
            if (other.Neurons.Count != this.Neurons.Count) return false;
            for (int i = 0; i < Neurons.Count; i++)
            {
                if (!other.Neurons[i].Equals(Neurons[i]))
                {
                    return false;
                }
            }
            return true;
        }

        public override int GetHashCode()
        {
            if (Neurons.Count == 0) return 0;
            return Neurons.Sum(n => n.GetHashCode()) / Neurons.Count;
        }

        public Matrix<double> BiasMatrix
        {
            get
            {
                if (Neurons == null) return null;
                if (!Neurons.Any()) return null;

                return Matrix<double>.Build.Dense(Neurons.Count, 1, Neurons.Select(n => n.Bias).ToArray());
            }
            set
            {
                var nIndex = 0;
                foreach (var bias in value.Storage.Enumerate())
                {
                    Neurons[nIndex].Bias = bias;
                    nIndex++;
                }
            }
        }


        public void Sigmoidize()
        {
            foreach (var neuron in Neurons)
            {
                neuron.Activation = Sigmoid(neuron.Activation);
            }
        }

        public void Reluize()
        {
            foreach (var neuron in Neurons)
            {
                neuron.Activation = Sigmoid(neuron.Activation);
            }
        }
        public static double Relu(double value)
        {
            if (value < 0) return 0;
            return value;
        }

        public static float Sigmoid(double value)
        {
            float k = (float)Exp(value);
            return k / (1.0f + k);
        }

        public static double Exp(double val)
        {
            long tmp = (long)(1512775 * val + 1072632447);
            return BitConverter.Int64BitsToDouble(tmp << 32);
        }

    }
}