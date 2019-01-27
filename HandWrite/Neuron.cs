using System;

namespace HandWrite
{
    public class Neuron
    {
        /// <summary>
        /// between 0.00 to 1.00, which is the brightness value
        /// </summary>
        public double Activation { get; set; }

        public double Bias { get; set; }


        public override bool Equals(object obj)
        {
            var other = (Neuron)obj;
            return (Math.Round(Activation, 2) == Math.Round(other.Activation, 2) && Math.Round(Bias, 4) == Math.Round(other.Bias, 4));
        }

        public override int GetHashCode()
        {
            return ((int)(Activation * 100)) + ((int)(Bias * 1000));
        }
    }
}